'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    deepseek_v3.py
'''
from .model_base import *
from .utils import WeightNameAdapter
from ..quantization import *
from .quantization_utils import *


class DeepSeek_v3(Model):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super().__init__("DeepSeek_v3", data_type, **kwargs)
        self.model.inputs.append(
            make_tensor("input_ids", np.empty(shape=(0, 0), dtype=np.int64)))
        self.model.inputs.append(
            make_tensor("attention_mask", np.empty(shape=(0, 0),
                                                   dtype=np.int64)))
        self.model.outputs.append(make_tensor("last_hidden_state"))
        self.is_generate = kwargs.get('is_generate', True)
        self._merge_gate_proj(torch_model)
        self.weight_real_names = set()
        for v in torch_model:
            self.weight_real_names.add(v)
        self._build_graph(self.model_config, derive_type)
        start_time = time.time()
        if not self.only_convert_lora:
            self._trans_weight(torch_model)
        self._trans_lora_weight(self._trans_weight)
        print("parse weight time: ", time.time() - start_time)

    def _merge_gate_proj(self, torch_model):
        """Merge gate_proj and up_proj into gate_up_proj for dense FFN,
        routed experts, and shared experts."""
        new_model = {}
        for key, val in torch_model.items():
            if key.find("gate_proj.weight") != -1:
                up_proj_key = key.replace("gate_proj", "up_proj")
                if up_proj_key in torch_model:
                    up_proj_val = torch_model[up_proj_key]
                    new_key = key.replace("gate_proj", "gate_up_proj")
                    tensor = torch.concat([val, up_proj_val]).cpu()
                    new_model[new_key] = tensor
        torch_model.update(new_model)

    def _build_graph(self, torch_cfg, derive_type):
        cfg = self.model.model_conf
        cfg.dtype = self.dtype

        cfg.ln_eps = torch_cfg.get('rms_norm_eps', 1e-6)
        cfg.num_heads = torch_cfg.get('num_attention_heads', 128)
        cfg.multi_query_group_num = torch_cfg.get('num_key_value_heads', 0)
        cfg.dec_layer = torch_cfg.get('num_hidden_layers', 61)
        cfg.hidden_size = torch_cfg.get('hidden_size', 7168)
        cfg.kv_channels = int(cfg.hidden_size / cfg.num_heads)
        cfg.activation = get_activation(torch_cfg.get('hidden_act', "silu"))
        cfg.size_per_head = torch_cfg.get('size_per_head', 128)
        cfg.intermediate_size = torch_cfg.get('intermediate_size', 18432)
        cfg.is_generate = self.is_generate

        # MOE config
        cfg.num_experts = torch_cfg.get('n_routed_experts', 256)
        cfg.num_experts_per_tok = torch_cfg.get('num_experts_per_tok', 8)

        # MLA config
        kv_lora_rank = torch_cfg.get('kv_lora_rank', 512)
        q_lora_rank = torch_cfg.get('q_lora_rank', 0) or 0  # None->0 for V2
        qk_nope_head_dim = torch_cfg.get('qk_nope_head_dim', 128)
        qk_rope_head_dim = torch_cfg.get('qk_rope_head_dim', 64)
        v_head_dim = torch_cfg.get('v_head_dim', 128)
        routed_scaling_factor = torch_cfg.get('routed_scaling_factor', 2.5)

        # Number of dense layers (first N layers use standard FFN, rest use MoE)
        first_k_dense_replace = torch_cfg.get('first_k_dense_replace', 3)
        moe_intermediate_size = torch_cfg.get('moe_intermediate_size', 2048)

        self.use_ep = torch_cfg.get('use_ep', False)

        # Check if e_score_correction_bias exists in model weights
        self.has_correction_bias = any(
            "e_score_correction_bias" in name
            for name in self.weight_real_names)

        # DeepSeek V2/V3 weight name mapping
        # Standard names used by WeightNameAdapter for regex matching
        weight_std_names = [
            # globals (index 0-2)
            "embed_tokens",                             # 0
            "norm.weight",                              # 1
            "lm_head",                                  # 2
        ]
        # Q projection weights depend on q_lora_rank
        if q_lora_rank > 0:
            weight_std_names += [
                "self_attn.q_a_proj.weight",                # 3
                "self_attn.q_a_layernorm.weight",           # 4
                "self_attn.q_b_proj.weight",                # 5
            ]
        else:
            weight_std_names += [
                "self_attn.q_proj.weight",                  # 3
            ]
        q_idx_end = len(weight_std_names)
        weight_std_names += [
            "self_attn.kv_a_proj_with_mqa.weight",      # q_idx_end+0
            "self_attn.kv_a_layernorm.weight",          # q_idx_end+1
            "self_attn.kv_b_proj.weight",               # q_idx_end+2
            "self_attn.o_proj.weight",                   # q_idx_end+3
            # per-layer norms
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            # dense FFN (first N layers) - merged gate+up
            "mlp.gate_up_proj.weight",
            "mlp.down_proj.weight",
            # shared expert
            "mlp.shared_experts.gate_up_proj.weight",
            "mlp.shared_experts.down_proj.weight",
            # MOE gate
            "mlp.gate.weight",
        ]
        # Index helpers
        kv_a_idx = q_idx_end
        o_proj_idx = q_idx_end + 3
        input_ln_idx = q_idx_end + 4
        post_ln_idx = q_idx_end + 5
        gate_up_proj_idx = q_idx_end + 6
        down_proj_idx = q_idx_end + 7
        shared_gate_up_idx = q_idx_end + 8
        shared_down_idx = q_idx_end + 9
        moe_gate_idx = q_idx_end + 10

        self.name_adapter = WeightNameAdapter(weight_std_names,
                                              self.weight_real_names,
                                              pattern_rules={
                                                  0: r"\b%s\b",
                                                  3: r"\blayers\.\d+\..*\b%s\b"
                                              })

        # Global weight name mapping
        self.weight_name_map = {
            "embedding.word_embeddings":
            self.name_adapter.fullname(weight_std_names[0]),
            "final.layernorm.gamma":
            self.name_adapter.fullname(weight_std_names[1]),
        }

        # Per-layer decoder name mapping
        decoder_name_map = {
            "attention.layernorm.gamma": weight_std_names[input_ln_idx],
            "attention.kv_a_proj.weight": weight_std_names[kv_a_idx],
            "attention.kv_a_norm.gamma": weight_std_names[kv_a_idx + 1],
            "attention.kv_b_proj.weight": weight_std_names[kv_a_idx + 2],
            "attention.output.dense.weight": weight_std_names[o_proj_idx],
            "ffn.layernorm.gamma": weight_std_names[post_ln_idx],
        }
        if q_lora_rank > 0:
            decoder_name_map["attention.q_a_proj.weight"] = weight_std_names[3]
            decoder_name_map["attention.q_a_norm.gamma"] = weight_std_names[4]
            decoder_name_map["attention.q_b_proj.weight"] = weight_std_names[5]
        else:
            decoder_name_map["attention.q_proj.weight"] = weight_std_names[3]

        for i in range(cfg.dec_layer):
            for key in decoder_name_map:
                real_name = decoder_name_map[key]
                if real_name in self.name_adapter.weight_name_segments:
                    self.weight_name_map["decoder.layer.{}.{}".format(
                        i, key)] = self.name_adapter.fullname(
                            real_name).format(i)

        # Dense FFN weights (first_k_dense_replace layers) - merged gate+up
        dense_ffn_name_map = {
            "ffn.gate_up_proj.weight": weight_std_names[gate_up_proj_idx],
            "ffn.down_proj.weight": weight_std_names[down_proj_idx],
        }
        for i in range(first_k_dense_replace):
            for key in dense_ffn_name_map:
                real_name = dense_ffn_name_map[key]
                if real_name in self.name_adapter.weight_name_segments:
                    self.weight_name_map["decoder.layer.{}.{}".format(
                        i, key)] = self.name_adapter.fullname(
                            real_name).format(i)

        # Shared expert weights (MoE layers only)
        shared_expert_name_map = {
            "shared_expert.gate_up_proj.weight": weight_std_names[shared_gate_up_idx],
            "shared_expert.down_proj.weight": weight_std_names[shared_down_idx],
            "mlp.gate.weight": weight_std_names[moe_gate_idx],
        }
        for i in range(first_k_dense_replace, cfg.dec_layer):
            for key in shared_expert_name_map:
                real_name = shared_expert_name_map[key]
                if real_name in self.name_adapter.weight_name_segments:
                    self.weight_name_map["decoder.layer.{}.{}".format(
                        i, key)] = self.name_adapter.fullname(
                            real_name).format(i)
            # e_score_correction_bias → gate GEMM bias
            if self.has_correction_bias:
                self.weight_name_map[
                    "decoder.layer.{}.mlp.gate.bias".format(i)] = \
                    "model.layers.{}.mlp.gate.e_score_correction_bias".format(i)

        # MOE expert weights (stacked)
        moe_name_map = {
            "gate_up_proj.weight": "gate_up_proj.weight",
            "down_proj.weight": "down_proj.weight",
        }
        for i in range(first_k_dense_replace, cfg.dec_layer):
            for key in moe_name_map:
                expert_name = "decoder.layer.{}.mlp.experts.{}".format(i, key)
                self.weight_name_map[expert_name] = [
                    "model.layers.{}.mlp.experts.{}.{}".format(i, j, key)
                    for j in range(cfg.num_experts)
                ]

        # Multi-GPU split map
        if self.multigpu_mode != 0:
            self.split_map = {}
            self.split_map["embedding.word_embeddings"] = VSPLIT
            for i in range(cfg.dec_layer):
                prefix = "decoder.layer.{}.".format(i)
                # MLA projections: split Q heads, keep KV unsplit (shared latent)
                self.split_map[prefix + "attention.q_b_proj.weight"] = VSPLIT
                self.split_map[prefix +
                               "attention.output.dense.weight"] = HSPLIT

                if i < first_k_dense_replace:
                    # Dense FFN layers (merged gate+up)
                    self.split_map[prefix +
                                   "ffn.gate_up_proj.weight"] = KVSPLIT
                    self.split_map[prefix +
                                   "ffn.down_proj.weight"] = HSPLIT
                else:
                    # MoE layers
                    self.split_map[prefix +
                                   "shared_expert.gate_up_proj.weight"] = KVSPLIT
                    self.split_map[prefix +
                                   "shared_expert.down_proj.weight"] = HSPLIT
                    if self.use_ep:
                        self.split_map[prefix +
                                       "mlp.experts.gate_up_proj.weight"] = EPSPLIT
                        self.split_map[prefix +
                                       "mlp.experts.down_proj.weight"] = EPSPLIT
                    else:
                        self.split_map[prefix +
                                       "mlp.experts.gate_up_proj.weight"] = BATCH_KVSPLIT
                        self.split_map[prefix +
                                       "mlp.experts.down_proj.weight"] = BATCH_HSPLIT

        # Quantization map
        if self.do_dynamic_quantize_convert is True:
            if self.quant_config is not None:
                if self.quant_config.quantize_mode in [
                        QuantizeConfig.QuantMode.A16W8
                ]:
                    self.quantize_map = {}
                    for i in range(cfg.dec_layer):
                        prefix = "decoder.layer.{}.".format(i)
                        self.quantize_map[prefix +
                                          "attention.q_a_proj.weight"] = 1
                        self.quantize_map[prefix +
                                          "attention.q_b_proj.weight"] = 1
                        self.quantize_map[prefix +
                                          "attention.kv_a_proj.weight"] = 1
                        self.quantize_map[prefix +
                                          "attention.kv_b_proj.weight"] = 1
                        self.quantize_map[prefix +
                                          "attention.output.dense.weight"] = 1
                        if i < first_k_dense_replace:
                            self.quantize_map[prefix +
                                              "ffn.gate_proj.weight"] = 1
                            self.quantize_map[prefix +
                                              "ffn.up_proj.weight"] = 1
                            self.quantize_map[prefix +
                                              "ffn.down_proj.weight"] = 1
                        else:
                            self.quantize_map[prefix +
                                              "shared_expert.gate_up_proj.weight"] = 1
                            self.quantize_map[prefix +
                                              "shared_expert.down_proj.weight"] = 1
                            self.quantize_map[prefix +
                                              "mlp.experts.gate_up_proj.weight"] = 1
                            self.quantize_map[prefix +
                                              "mlp.experts.down_proj.weight"] = 1
                else:
                    raise RuntimeError(
                        "quantize mode {} is not supported.".format(
                            self.quant_config.quantize_mode))
            else:
                raise RuntimeError("quantize config is None")

        do_binary_add_fused = self.do_binary_add_fused
        if self.do_dynamic_quantize_convert or self.gen_lora_op:
            do_binary_add_fused = False
        self._make_lora_split_map()
        self._make_lora_quant_map()

        ##############################################################################################
        # Build computation graph
        ##############################################################################################
        self.model.graph_names.extend(["decoder"])
        graph = self.model.graphs["decoder"]
        mask = TransMask(
            "transmask",
            self.model.inputs[1],
            {"sequence_mask": True},
        )()
        embedding = EmbeddingT5("embedding", self.model.inputs[0],
                                {"token_embedding": False})()
        graph.ops.extend([mask, embedding])
        if self.multigpu_mode != 0:
            all_gather_embedding = AllGather("all_gather_embedding",
                                             embedding.outputs[0])()
            graph.ops.append(all_gather_embedding)
            rich_embedding = RichEmbedding(
                "rich_embedding",
                [self.model.inputs[0], all_gather_embedding.outputs[0]])()
            graph.ops.append(rich_embedding)
        else:
            rich_embedding = RichEmbedding(
                "rich_embedding",
                [self.model.inputs[0], embedding.outputs[0]])()
            graph.ops.append(rich_embedding)

        for i in range(cfg.dec_layer):
            prefix = "decoder.layer.{}.".format(i)
            attn_op_list = []
            ffn_op_list = []

            ####################################################################
            # Attention block: MLA
            ####################################################################
            first_ln = LayerNormNoBeta(
                prefix + "attention.layernorm",
                graph.ops[-1].outputs[0],
                {"eps": cfg.ln_eps},
            )()

            # MLA attention: the operator handles all internal projections
            # (q_a_proj, q_a_norm, q_b_proj, kv_a_proj, kv_a_norm, kv_b_proj)
            # as well as RoPE application and attention computation.
            mla_attn = MLAAttention(
                prefix + "attention",
                [first_ln.outputs[0], mask.outputs[0]],
                {
                    "num_heads": cfg.num_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "q_lora_rank": q_lora_rank,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "v_head_dim": v_head_dim,
                },
            )()

            attn_out_gemm = self.make_gemm_op(
                prefix + "attention.output.dense", mla_attn.outputs[0], {
                    "with_bias": False,
                })()

            all_reduce_attention = AllReduce(
                prefix + "attention.all_reduce_attention",
                attn_out_gemm.outputs[0])()

            attn_add = Binary(
                prefix + "attention_add",
                [attn_out_gemm.outputs[0], graph.ops[-1].outputs[0]],
                {"binary_type": ADD},
            )()

            attn_op_list = [
                first_ln, mla_attn, attn_out_gemm,
                all_reduce_attention, attn_add
            ]

            ####################################################################
            # FFN block: Dense (first 3 layers) or MoE (layers 3+)
            ####################################################################
            ffn_ln = LayerNormNoBeta(prefix + "ffn.layernorm",
                                    attn_add.outputs[0],
                                    {"eps": cfg.ln_eps})()
            ffn_input = ffn_ln.outputs[0]

            if i < first_k_dense_replace:
                # Dense FFN: merged gate_up_proj (SwiGLU) -> down_proj
                ffn_gate_up = self.make_gemm_op(
                    prefix + "ffn.gate_up_proj", ffn_input,
                    {"with_bias": False})()
                ffn_act = UnaryGLU(
                    prefix + "ffn.act_mul",
                    [ffn_gate_up.outputs[0]],
                    {"unary_type": cfg.activation},
                )()
                ffn_down = self.make_gemm_op(
                    prefix + "ffn.down_proj", ffn_act.outputs[0],
                    {"with_bias": False})()
                all_reduce_ffn = AllReduce(
                    prefix + "all_reduce_ffn", ffn_down.outputs[0])()
                final_add = Binary(
                    prefix + "final_add",
                    [ffn_down.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()
                ffn_op_list = [
                    ffn_ln, ffn_gate_up, ffn_act, ffn_down,
                    all_reduce_ffn, final_add
                ]
            else:
                # MoE FFN: gate -> MOE experts + shared expert (no sigmoid gate)
                mlp_gate = self.make_gemm_op(
                    prefix + "mlp.gate", ffn_input,
                    {"with_bias": self.has_correction_bias})()

                # DeepSeek V3 uses grouped routing:
                #   routing_mode=1: sigmoid + grouped top-k
                #   num_group=8: 256 experts / 8 groups = 32 per group
                #   top_k_group=4: select top 4 groups
                #   top_k=8: select 2 experts per group = 8 total
                scoring_func = torch_cfg.get('scoring_func', 'sigmoid')
                n_group = torch_cfg.get('n_group', 8)
                topk_group = torch_cfg.get('topk_group', 4)
                # routing_mode: 0=softmax+topk (V2), 1=sigmoid+grouped topk (V3)
                routing_mode = 1 if scoring_func == 'sigmoid' else 0
                mlp_moe = MOE(
                    prefix + "mlp.experts",
                    [ffn_input, mlp_gate.outputs[0]],
                    {
                        "num_experts": cfg.num_experts,
                        "num_experts_per_tok": cfg.num_experts_per_tok,
                        "use_ep": self.use_ep,
                        "routing_mode": routing_mode,
                        "num_group": n_group,
                        "top_k_group": topk_group,
                        "routed_scaling_factor": routed_scaling_factor,
                    }
                )()
                all_reduce_moe = AllReduce(
                    prefix + "attention.all_reduce_moe",
                    mlp_moe.outputs[0])()

                # Shared expert (no sigmoid gate, unlike Qwen MOE)
                shared_expert_gate_up_proj = self.make_gemm_op(
                    prefix + "shared_expert.gate_up_proj",
                    ffn_ln.outputs[0],
                    {"with_bias": False})()
                shared_expert_unary_glu = UnaryGLU(
                    prefix + "shared_expert_act_mul",
                    [shared_expert_gate_up_proj.outputs[0]],
                    {"unary_type": cfg.activation},
                )()
                shared_expert_down_proj = self.make_gemm_op(
                    prefix + "shared_expert.down_proj",
                    shared_expert_unary_glu.outputs[0],
                    {"with_bias": False})()
                all_reduce_shared_expert = AllReduce(
                    prefix + "all_reduce_shared_expert",
                    shared_expert_down_proj.outputs[0])()

                # DeepSeek V3: no shared_expert_gate (unlike Qwen MOE)
                # Just add routed expert output + shared expert output
                expert_add = Binary(
                    prefix + "expert_add",
                    [mlp_moe.outputs[0], shared_expert_down_proj.outputs[0]],
                    {"binary_type": ADD},
                )()

                final_add = Binary(
                    prefix + "final_add",
                    [expert_add.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()

                ffn_op_list = [
                    ffn_ln, mlp_gate, mlp_moe, all_reduce_moe,
                    shared_expert_gate_up_proj, shared_expert_unary_glu,
                    shared_expert_down_proj, all_reduce_shared_expert,
                    expert_add, final_add
                ]

            graph.ops.extend(attn_op_list + ffn_op_list)

        # Final layernorm
        final_layernorm = LayerNormNoBeta("final.layernorm",
                                          graph.ops[-1].outputs[0],
                                          {"eps": cfg.ln_eps})()
        graph.ops.append(final_layernorm)
        graph.ops[-1].outputs[0].name = "last_hidden_state"

        # Quantize
        if self.do_dynamic_quantize_convert:
            for op in graph.ops:
                quantize_op(op, self.quant_config, self.quantize_map)

        ##############################################################################################
        # Output layer
        ##############################################################################################
        if derive_type is None:
            raise RuntimeError(
                "derive type [{}] is not supported.".format(derive_type))
        elif derive_type == "lmhead":
            self._add_layer("lmhead", graph, graph.ops[-1].outputs[0])
            self.weight_name_map.update({
                "lm_head.weight":
                self.name_adapter.fullname(weight_std_names[2]),
            })
            graph.ops[-1].outputs[0].name = "logits"
            self.model.outputs[0].CopyFrom(graph.ops[-1].outputs[0])
        else:
            raise RuntimeError(
                "derive type [{}] is not supported.".format(derive_type))

        ##############################################################################################
        # Generation graphs
        ##############################################################################################
        if self.is_generate:
            self.model.graph_names.insert(0, "pre_graph")
            self.model.graph_names.append("gen_graph")
            gen_graph = self.model.graphs["gen_graph"]
            self.model.graph_names.append("post_graph")
            self.model.outputs[0].CopyFrom(
                make_tensor("generated_ids",
                            np.empty(shape=(0, 0), dtype=np.int64)))
            pre_graph = self.model.graphs["pre_graph"]
            preprocess_ids = PreProcessId(
                "preprocess_id",
                self.model.inputs[0],
            )()
            update_id_first = UpdateId("update_id_first",
                                       preprocess_ids.outputs[0])()
            pre_graph.ops.extend(
                [preprocess_ids, update_id_first, graph.ops[0]])
            del graph.ops[0]

            for op in graph.ops:
                if op.op_type == "EmbeddingT5":
                    op.inputs[0].CopyFrom(preprocess_ids.outputs[0])
                elif op.op_type == "MLAAttention":
                    op.op_type = "DecOptMLA"

            gen_op = GenerateOp(
                "generate",
                [graph.ops[-1].outputs[0], preprocess_ids.outputs[1]],
            )()
            update_id = UpdateId(
                "update_id",
                [preprocess_ids.outputs[0], gen_op.outputs[1]])()
            postprocess_ids = PostProcessId(
                "postprocess_id",
                [update_id.outputs[0], gen_op.outputs[2]])()
            for op in graph.ops:
                if op.op_type == "DecOptMLA":
                    op.inputs.append(gen_op.outputs[1])
            gen_op.outputs[0].CopyFrom(preprocess_ids.outputs[0])
            gen_graph.ops.extend([gen_op, update_id])

            post_graph = self.model.graphs["post_graph"]
            post_graph.ops.append(postprocess_ids)

    def _trans_weight(self, torch_weight, lora_name=None):
        if not lora_name:
            weights_path = self.weights_path
            weight_name_map = self.weight_name_map
            split_map = self.split_map
            sparse_map = self.sparse_map
            quantize_map = self.quantize_map
        else:
            weights_path = super()._get_lora_path(lora_name)
            with open(weights_path, "w") as f:
                f.truncate(0)
            self.lora_weight_name_map[lora_name] = super(
            )._make_lora_weight_name_map(list(torch_weight.keys()),
                                         self.weight_name_map)
            weight_name_map = self.lora_weight_name_map[lora_name]
            for internal_name in list(weight_name_map):
                if isinstance(weight_name_map[internal_name], list):
                    for e_name in weight_name_map[internal_name]:
                        if torch_weight.get(e_name) is None:
                            print(
                                f"{e_name} not found in lora weight of {lora_name}"
                            )
                            if weight_name_map.get(internal_name):
                                del weight_name_map[internal_name]
                elif torch_weight.get(
                        weight_name_map[internal_name]) is None:
                    print(
                        f"{internal_name} not found in lora weight of {lora_name}"
                    )
                    del weight_name_map[internal_name]
            split_map = self.lora_split_map[lora_name]
            sparse_map = {}
            quantize_map = self.lora_quantize_map[lora_name]

        self_dtype_str = [
            k for k, v in Model.dtype_dict.items() if v == self.dtype
        ][0]
        for key, torch_name in weight_name_map.items():
            if isinstance(torch_name, list):
                if "experts" in key:
                    tensor = (torch.stack(
                        [torch.permute(torch_weight[name], (1, 0)).contiguous()
                         for name in torch_name]).cpu())
                else:
                    tensor = (torch.concat(
                        [torch_weight[name] for name in torch_name]).cpu())
                    if key.find("weight") != -1:
                        tensor = torch.permute(tensor, (1, 0)).contiguous()
            else:
                tensor = torch_weight[torch_name].cpu()
                if key.find("weight") != -1:
                    tensor = torch.permute(tensor, (1, 0)).contiguous()
            self.validate_weight_dtype(key, tensor, self_dtype_str)
            mode = DENSE if key not in sparse_map else sparse_map[key]
            split_mode = NOSPLIT if key not in split_map else split_map[key]
            if split_mode != GROUP_VSPLIT:
                group_list = []
            else:
                group_list = [
                    self.model.model_conf.num_heads *
                    self.model.model_conf.kv_channels,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.kv_channels,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.kv_channels
                ]
            quantize_mode = (False if lora_name or key not in quantize_map
                             else quantize_map[key])
            if quantize_mode is False:
                save_torch_to_allsparky(weights_path, key, tensor, mode,
                                        split_mode, group_list)
            else:
                if self.quant_config.quantize_mode == \
                        QuantizeConfig.QuantMode.A16W8:
                    qdata = None
                    scale = None
                    zero_point = None
                    if "experts" in key:
                        q_data_list = []
                        scale_list = []
                        zero_list = []
                        for data in tensor:
                            sub_qdata, sub_scale, sub_zero_point = \
                                quantize_gemm_weight_a16w8_torch(
                                    data, self.quant_config)
                            q_data_list.append(sub_qdata)
                            scale_list.append(sub_scale)
                            zero_list.append(sub_zero_point)
                        qdata = torch.stack(q_data_list, dim=0)
                        scale = torch.stack(scale_list, dim=0)
                        zero_point = torch.stack(zero_list, dim=0)
                    else:
                        qdata, scale, zero_point = \
                            quantize_gemm_weight_a16w8_torch(
                                tensor, self.quant_config)
                    save_torch_to_allsparky(weights_path, key, qdata, mode,
                                            split_mode, group_list)
                    if (self.quant_config.extra_option["SubChannel"] is True
                            or (split_mode != HSPLIT
                                and split_mode != BATCH_HSPLIT)):
                        save_torch_to_allsparky(weights_path, key + ".scale",
                                                scale, mode, split_mode,
                                                group_list)
                        save_torch_to_allsparky(weights_path,
                                                key + ".zero_point",
                                                zero_point, mode, split_mode,
                                                group_list)
                    else:
                        save_torch_to_allsparky(weights_path, key + ".scale",
                                                scale, mode, NOSPLIT,
                                                group_list)
                        save_torch_to_allsparky(weights_path,
                                                key + ".zero_point",
                                                zero_point, mode, NOSPLIT,
                                                group_list)
            if isinstance(torch_name, list):
                for name in torch_name:
                    torch_weight[name] = torch.Tensor(0)
            else:
                torch_weight[torch_name] = torch.Tensor(0)
        set_global_header(weights_path)
