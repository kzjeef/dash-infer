'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    qwen_v30.py
'''
from .model_base import *
from .utils import WeightNameAdapter
from ..quantization import *
from .quantization_utils import *


class Qwen_v30(Model):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super().__init__("Qwen_v30", data_type, **kwargs)
        self.model.inputs.append(
            make_tensor("input_ids", np.empty(shape=(0, 0), dtype=np.int64)))
        self.model.inputs.append(
            make_tensor("attention_mask", np.empty(shape=(0, 0),
                                                   dtype=np.int64)))
        self.model.outputs.append(make_tensor("last_hidden_state"))
        self.is_generate = kwargs.get('is_generate', True)
        self.weight_real_names = set()
        self.covert_namespace_qweight_to_weight(torch_model)
        for v in torch_model:
            self.weight_real_names.add(v)
        self._build_graph(self.model_config, derive_type)
        start_time = time.time()
        if not self.only_convert_lora:
            self._trans_weight(torch_model)
        self._trans_lora_weight(self._trans_weight)
        print("parse weight time: ", time.time() - start_time)

    def _build_graph(self, torch_cfg, derive_type):
        cfg = self.model.model_conf
        cfg.dtype = self.dtype

        cfg.ln_eps = torch_cfg.get('rms_norm_eps', 1e-6)
        cfg.num_heads = torch_cfg.get('num_attention_heads', 12)
        cfg.multi_query_group_num = torch_cfg.get('num_key_value_heads', 0)
        if (cfg.multi_query_group_num == 0):
            cfg.multi_query_group_num = cfg.num_heads
        cfg.dec_layer = torch_cfg.get('num_hidden_layers', 12)
        hidden_size_ = torch_cfg.get('hidden_size', 4096)
        # cfg.head_dim = int(hidden_size_ / cfg.num_heads)
        cfg.kv_channels = int(hidden_size_ / cfg.num_heads)
        cfg.activation = get_activation(torch_cfg.get('hidden_act', "silu"))
        cfg.size_per_head = torch_cfg.get('size_per_head', cfg.kv_channels)
        cfg.intermediate_size = torch_cfg.get('intermediate_size', 0)
        cfg.is_generate = self.is_generate
        rope_scaling = torch_cfg.get('rope_scaling',{})
        if rope_scaling is None:
            rope_scaling = {}
        # daoxian added for span version
        cfg.dtype = self.dtype

        weight_std_names = [
            # globals
            "embed_tokens",  #0
            "norm.weight",
            "lm_head",  #2
            # layers
            "q_proj.weight",  #3
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "input_layernorm.weight",  #7
            "post_attention_layernorm.weight",  #8
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "rotary_emb.inv_freq",  #12
            "q_norm.weight",  #13
            "k_norm.weight",
        ]

        self.name_adapter = WeightNameAdapter(weight_std_names,
                                              self.weight_real_names,
                                              pattern_rules={
                                                  0: r"\b%s\b",
                                                  3: r"\blayers\.\d+\..*\b%s\b"
                                              })
        self.weight_name_map = {
            "embedding.word_embeddings":
            self.name_adapter.fullname(weight_std_names[0]),
            "final.layernorm.gamma":
            self.name_adapter.fullname(weight_std_names[1]),
        }
        print("torch_cfg.embed_proj=",torch_cfg.get('embed_proj', False))
        if torch_cfg.get('embed_proj', False):
            self.weight_name_map["embed_proj.weight"] = "thinker_to_talker_proj.weight"
            self.weight_name_map["embed_proj.bias"] = "thinker_to_talker_proj.bias"
        decoder_name_map = {
            "attention.layernorm.gamma":
            weight_std_names[7],
            "attention.self.weight":
            [weight_std_names[3], weight_std_names[4], weight_std_names[5]],
            # "attention.self.bias":
            # [weight_std_names[13], weight_std_names[14], weight_std_names[15]],
            "attention.output.dense.weight":
            weight_std_names[6],
            "ffn.layernorm.gamma":
            weight_std_names[8],
            "ffn.intermediate.dense.weight":
            weight_std_names[9],
            "ffn.linear.dense.weight":
            weight_std_names[10],
            "ffn.output.dense.weight":
            weight_std_names[11],
            "rotary.inv_freq":
            weight_std_names[12],
            "attention.qknorm.q_norm.gamma":
            weight_std_names[13],
            "attention.qknorm.k_norm.gamma":
            weight_std_names[14],
                
        }
        for i in range(cfg.dec_layer):
            for key in decoder_name_map:
                real_name = decoder_name_map[key]
                if isinstance(real_name, list):
                    # print(f"convert: decode layer: {'decoder.layer.{}.{}'.format(i, key)}")
                    self.weight_name_map["decoder.layer.{}.{}".format(
                        i, key)] = [(self.name_adapter.fullname(v).format(i))
                                    for v in real_name]
                else:
                    if real_name in self.name_adapter.weight_name_segments:
                        self.weight_name_map["decoder.layer.{}.{}".format(
                            i, key)] = self.name_adapter.fullname(
                                real_name).format(i)
        if self.multigpu_mode != 0:
            self.split_map = {}
            self.split_map["embedding.word_embeddings"] = VSPLIT
            for i in range(cfg.dec_layer):
                prefix = "decoder.layer.{}.".format(i)
                if cfg.multi_query_group_num == cfg.num_heads:
                    self.split_map[prefix + "attention.self.weight"] = QKVSPLIT
                    self.split_map[prefix + "attention.self.bias"] = QKVSPLIT
                else:
                    self.split_map[prefix +
                                   "attention.self.weight"] = GROUP_VSPLIT
                    self.split_map[prefix +
                                   "attention.self.bias"] = GROUP_VSPLIT
                self.split_map[prefix +
                               "attention.output.dense.weight"] = HSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense.weight"] = VSPLIT
                self.split_map[prefix + "ffn.linear.dense.weight"] = VSPLIT
                self.split_map[prefix + "ffn.output.dense.weight"] = HSPLIT
        if self.do_dynamic_quantize_convert is True:
            if self.quant_config != None:
                print(self.quant_config.quantize_mode)
                if self.quant_config.quantize_mode in [
                        QuantizeConfig.QuantMode.A16W8,
                        QuantizeConfig.QuantMode.A16W4,
                        QuantizeConfig.QuantMode.A8W8,
                        QuantizeConfig.QuantMode.FP8A8W8
                ]:
                    self.quantize_map = {}
                    for i in range(cfg.dec_layer):
                        prefix = "decoder.layer.{}.".format(i)
                        self.quantize_map[prefix + "attention.self.weight"] = 1
                        self.quantize_map[prefix +
                                          "attention.output.dense.weight"] = 1
                        self.quantize_map[prefix +
                                          "ffn.intermediate.dense.weight"] = 1
                        self.quantize_map[prefix +
                                          "ffn.linear.dense.weight"] = 1
                        self.quantize_map[prefix +
                                          "ffn.output.dense.weight"] = 1
                else:
                    raise RuntimeError(
                        "quantize mode {} is not supported.".format(
                            self.quant_config.quantize_mode))
            else:
                raise RuntimeError("quantize config is None")
        # fused binary_add is not supported in dynamic quantized model or lora mode
        do_binary_add_fused = self.do_binary_add_fused
        if self.do_dynamic_quantize_convert or self.gen_lora_op:
            do_binary_add_fused = False
        self._make_lora_split_map()
        self._make_lora_quant_map()
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
        if torch_cfg.get('embed_proj', False):
            embed_proj = self.make_gemm_op(
                "embed_proj",
                graph.ops[-1].outputs[0],
                {
                    "with_bias": True,
                },
            )()
            graph.ops.append(embed_proj)
        for i in range(cfg.dec_layer):
            prefix = "decoder.layer.{}.".format(i)
            # attention
            first_ln = LayerNormNoBeta(
                prefix + "attention.layernorm",
                graph.ops[-1].outputs[0],
                {"eps": cfg.ln_eps},
            )()
            attn_self_gemm = self.make_gemm_op(
                prefix + "attention.self", first_ln.outputs[0], {
                    "with_bias": False,
                    
                })()
            qk_norm = QKLayerNormNoBeta(
                prefix + "attention.qknorm",
                attn_self_gemm.outputs[0],
                {"eps": cfg.ln_eps,"num_heads": cfg.num_heads,
                "size_per_head": cfg.size_per_head,
                "multi_query_group_num": cfg.multi_query_group_num,},
            )()
            rotary_embedding = None
            attn = None
            rotary_attributes = {"num_heads": cfg.num_heads}
            if self.use_dynamic_ntk and hasattr(self, "model_sequence_length"):
                rotary_attributes["ntk_model_embed"] = int(
                    self.model_sequence_length)
            if hasattr(self, "seqlen_extrapolation"):
                rotary_attributes["seqlen_extrapolation"] = float(
                    self.seqlen_extrapolation)
            if hasattr(self, "rotary_base"):
                rotary_attributes["rotary_base"] = float(self.rotary_base)
            if self.use_logn_attn and hasattr(self, "model_sequence_length"):
                rotary_attributes["logn_model_embedding"] = int(
                    self.model_sequence_length)
            if cfg.multi_query_group_num != cfg.num_heads:
                rotary_attributes["multi_query_group_num"] = int(
                    cfg.multi_query_group_num)
            if rope_scaling !={}:
                if "mrope_section" in rope_scaling:
                    # rotary_attributes["rope_type"] = get_invfreq_type(rope_scaling["type"])
                    mrope_section_list = rope_scaling["mrope_section"]
                    # [16,24,24]
                    rotary_attributes["mrope_section_size"] = len(mrope_section_list)
                    for index, value in enumerate(mrope_section_list):
                        attr_name = "mrope_section_" + str(index)
                        rotary_attributes[attr_name] = value
                        
                else:
                    rotary_attributes["original_max_position_embeddings"] = int(
                        rope_scaling["original_max_position_embeddings"])
                    rotary_attributes["factor"] = int(rope_scaling["factor"])
                    rotary_attributes["invfreq_type"] = get_invfreq_type(rope_scaling["type"])
            rotary_embedding = Rotary(
                prefix + "rotary",
                [qk_norm.outputs[0], mask.outputs[1]],
                rotary_attributes,
            )()
            if cfg.multi_query_group_num == cfg.num_heads:
                attn = MultiHeadAttention(
                    prefix + "attention",
                    [rotary_embedding.outputs[0], mask.outputs[0]],
                    {"num_heads": cfg.num_heads},
                )()
            else:
                attn = MultiQueryAttention(
                    prefix + "attention",
                    [rotary_embedding.outputs[0], mask.outputs[0]],
                    {
                        "num_heads": cfg.num_heads,
                        "size_per_head": cfg.size_per_head,
                        "multi_query_group_num": cfg.multi_query_group_num,
                        "multigpu": 1,
                    },
                )()
            attn_op_list = []
            ffn_op_list = []
            ffn_ln = None
            if do_binary_add_fused:
                attn_out_gemm = self.make_gemm_op(
                    prefix + "attention.output.dense",
                    [attn.outputs[0], graph.ops[-1].outputs[0]], {
                        "with_bias": False,
                        "binary_type": ADD
                    })()
                attn_op_list = [
                    first_ln, attn_self_gemm, qk_norm,rotary_embedding, attn,
                    attn_out_gemm
                ]
                # ffn
                ffn_ln = LayerNormNoBeta(prefix + "ffn.layernorm",
                                         attn_out_gemm.outputs[0],
                                         {"eps": cfg.ln_eps})()
            else:
                attn_out_gemm = self.make_gemm_op(
                    prefix + "attention.output.dense", attn.outputs[0], {
                        "with_bias": False,
                        
                    })()
                attn_add = Binary(
                    prefix + "attention_add",
                    [attn_out_gemm.outputs[0], graph.ops[-1].outputs[0]],
                    {"binary_type": ADD},
                )()
                attn_op_list = [
                    first_ln, attn_self_gemm, qk_norm, rotary_embedding, attn,
                    attn_out_gemm, attn_add
                ]
                # ffn
                ffn_ln = LayerNormNoBeta(prefix + "ffn.layernorm",
                                         attn_add.outputs[0],
                                         {"eps": cfg.ln_eps})()
            ffn_intermediate = self.make_gemm_op(
                prefix + "ffn.intermediate.dense",
                ffn_ln.outputs[0],
                {
                    "activation": cfg.activation,
                    "with_bias": False,
                    
                },
            )()
            ffn_linear = self.make_gemm_op(
                prefix + "ffn.linear.dense",
                ffn_ln.outputs[0],
                {
                    "with_bias": False,
                    
                },
            )()
            ffn_mul = Binary(
                prefix + "ffn.mul",
                [ffn_intermediate.outputs[0], ffn_linear.outputs[0]],
                {"binary_type": MUL},
            )()
            if do_binary_add_fused:
                ffn_out = self.make_gemm_op(
                    prefix + "ffn.output.dense",
                    [ffn_mul.outputs[0], attn_out_gemm.outputs[0]], {
                        "binary_type": ADD,
                        "with_bias": False,
                        
                    })()
                ffn_op_list = [
                    ffn_ln, ffn_intermediate, ffn_linear, ffn_mul, ffn_out
                ]
                if self.multigpu_mode != 0:
                    all_reduce_attention = AllReduce(
                        prefix + "attention.all_reduce_attention",
                        attn_out_gemm.outputs[0])()
                    all_reduce_ffn = AllReduce(
                        prefix + "attention.all_reduce_ffn",
                        ffn_out.outputs[0])()
                    attn_op_list.append(all_reduce_attention)
                    ffn_op_list.append(all_reduce_ffn)
            else:
                ffn_out = self.make_gemm_op(
                    prefix + "ffn.output.dense", ffn_mul.outputs[0], {
                        "with_bias": False,
                        
                    })()
                final_add = Binary(
                    prefix + "final_add",
                    [ffn_out.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()
                ffn_op_list = [
                    ffn_ln, ffn_intermediate, ffn_linear, ffn_mul, ffn_out,
                    final_add
                ]
                if self.multigpu_mode != 0:
                    all_reduce_attention = AllReduce(
                        prefix + "attention.all_reduce_attention",
                        attn_out_gemm.outputs[0])()
                    all_reduce_ffn = AllReduce(
                        prefix + "attention.all_reduce_ffn",
                        ffn_out.outputs[0])()
                    attn_op_list.insert(-1, all_reduce_attention)
                    ffn_op_list.insert(-1, all_reduce_ffn)
            # final
            graph.ops.extend(attn_op_list + ffn_op_list)

        #deocder over
        final_layernorm = LayerNormNoBeta("final.layernorm",
                                          graph.ops[-1].outputs[0],
                                          {"eps": cfg.ln_eps})()
        graph.ops.append(final_layernorm)
        graph.ops[-1].outputs[0].name = "last_hidden_state"
        # Quantize
        if self.do_dynamic_quantize_convert:
            for op in graph.ops:
                quantize_op(op, self.quant_config, self.quantize_map, self.weight_name_map)
        ##############################################################################################
        if derive_type == None:
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
            #########################################################
            for op in graph.ops:
                if op.op_type == "EmbeddingT5":
                    # op.op_type = "DecOptEmbedding"
                    op.inputs[0].CopyFrom(preprocess_ids.outputs[0])
                elif op.op_type == "MultiHeadAttention":
                    op.op_type = "DecOptMHA"
                elif op.op_type == "MultiQueryAttention":
                    op.op_type = "DecOptMQA"
            gen_op = GenerateOp(
                "generate",
                [graph.ops[-1].outputs[0], preprocess_ids.outputs[1]],
            )()
            update_id = UpdateId(
                "update_id", [preprocess_ids.outputs[0], gen_op.outputs[1]])()
            postprocess_ids = PostProcessId(
                "postprocess_id", [update_id.outputs[0], gen_op.outputs[2]])()
            for op in graph.ops:
                if op.op_type == "DecOptMHA" or op.op_type == "DecOptMQA" or op.op_type == "DecOptMQAI8Cache":
                    op.inputs.append(gen_op.outputs[1])
            gen_op.outputs[0].CopyFrom(preprocess_ids.outputs[0])
            gen_graph.ops.extend([gen_op, update_id])
            #########################################################
            post_graph = self.model.graphs["post_graph"]
            post_graph.ops.append(postprocess_ids)

    def _trans_weight(self, torch_weight, lora_name=None):
        if not lora_name:
            weights_path = self.weights_path
            weight_name_map = self.weight_name_map
            split_map = self.split_map
            sparse_map = self.sparse_map
            quantize_map = self.quantize_map
        else:  # for LoRA
            weights_path = super()._get_lora_path(lora_name)
            with open(weights_path, "w") as f:
                f.truncate(0)
            # 生成并校验lora的name map
            self.lora_weight_name_map[lora_name] = super(
            )._make_lora_weight_name_map(list(torch_weight.keys()),
                                         self.weight_name_map)
            weight_name_map = self.lora_weight_name_map[lora_name]
            for internal_name in list(weight_name_map):
                if isinstance(weight_name_map[internal_name], list):
                    for e_name in weight_name_map[internal_name]:
                        if torch_weight.get(e_name) == None:
                            print(
                                f"{e_name} not found in lora weight of {lora_name}"
                            )
                            if weight_name_map.get(internal_name):
                                del weight_name_map[internal_name]
                elif torch_weight.get(
                        weight_name_map[internal_name]
                ) == None:  # ignore LoRA possible nonexistent keys
                    print(
                        f"{internal_name} not found in lora weight of {lora_name}"
                    )
                    del weight_name_map[internal_name]
            split_map = self.lora_split_map[lora_name]
            #sparse_map = self.lora_sparse_map[lora_name]
            sparse_map = {}
            quantize_map = self.lora_quantize_map[lora_name]

        self_dtype_str = [
            k for k, v in Model.dtype_dict.items() if v == self.dtype
        ][0]
        for key, torch_name in weight_name_map.items():
            if isinstance(torch_name, list):  # attention qkv weights
                tensor = (torch.concat(
                    [torch_weight[name] for name in torch_name]).cpu())
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
                    self.model.model_conf.size_per_head,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.size_per_head,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.size_per_head
                ]
            quantize_mode = False if lora_name or key not in quantize_map else quantize_map[
                key]  # lora量化的时候， 去掉lora_name判断即可
            if quantize_mode == False:
                save_torch_to_allsparky(weights_path, key, tensor, mode,
                                        split_mode, group_list)
            else:
                if self.quant_config.quantize_mode in [
                        QuantizeConfig.QuantMode.A16W8,
                        QuantizeConfig.QuantMode.A16W4,
                        QuantizeConfig.QuantMode.A8W8,
                        QuantizeConfig.QuantMode.FP8A8W8
                ]:
                    qdata = None
                    scale = None
                    zero_point = None
                    if isinstance(torch_name, list):
                        # quant separated qkv
                        qdata, scale, zero_point = self.quant_qkv_seperated_weight(qdata, scale, tensor, torch_name,
                                                                                   torch_weight, zero_point)
                    else:
                        # quant qkv is not separated
                        qdata, scale, zero_point = quantize_gemm_weight_a16wX_torch(
                            tensor, self.quant_config,
                            [self.name_adapter.origname(torch_name), torch_weight])
                    # Check whether splitting is possible under the SubChannel settings
                    if (self.quant_config.extra_option["SubChannel"] == True) and (split_mode == HSPLIT):
                        K = qdata.shape[0]
                        group_size = get_groupsize_by_quant_config(self.quant_config, torch_name)
                        if K % group_size != 0:
                            raise ValueError(
                                f"SubChannel: the model is not splittable under current Groupsize value {group_size}. "
                                f"Weight Tensor Name: {key}, Split Mode: HSPLIT, Shape: {tensor.shape}"
                            )
                    if (group_list!=[] and self.quant_config.quantize_mode == QuantizeConfig.QuantMode.A16W4):
                        save_torch_to_allsparky(weights_path, key, qdata, mode,
                                            split_mode, [x//2 for x in group_list ])
                    else:
                        save_torch_to_allsparky(weights_path, key, qdata, mode,
                                                split_mode, group_list)
                    if (self.quant_config.extra_option["SubChannel"] == True or split_mode != HSPLIT) and self.quant_config.quantize_mode != QuantizeConfig.QuantMode.FP8A8W8:
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

    def quant_qkv_seperated_weight(self, qdata, scale, tensor, torch_name, torch_weight, zero_point):
        if self.quant_config.extra_option.get("AdaptedQuantMethod") in ["GPTQ", "GPTQ_NO_PACK"]:
            sub_qdata = {}
            sub_scale = {}
            sub_zero_point = {}
            for name in torch_name:
                sub_qdata[name], sub_scale[name], sub_zero_point[name] = quantize_gemm_weight_a16wX_torch(
                    tensor, self.quant_config,
                    [self.name_adapter.origname(name), torch_weight])
            qdata = (torch.concat(
                [sub_qdata[name] for name in torch_name], dim=1).cpu())
            scale = (torch.concat(
                [sub_scale[name] for name in torch_name], dim=1).cpu())
            zero_point = (torch.concat(
                [sub_zero_point[name] for name in torch_name], dim=1).cpu())
        else:
            qdata, scale, zero_point = quantize_gemm_weight_a16wX_torch(
                tensor, self.quant_config, None)
            ###orignal a16w8
        return qdata, scale, zero_point
