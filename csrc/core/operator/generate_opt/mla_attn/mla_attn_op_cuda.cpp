/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mla_attn_op_cuda.cpp
 */

#ifdef ENABLE_CUDA
#include "mla_attn_op_cuda.h"

#include <cmath>

#include "core/kernel/kernel.h"
#include "utility/datatype_dispatcher.h"

namespace allspark {

void MLAAttnOpCUDA::computeInvFreq(float rope_base) {
  int freq_size = qk_rope_head_dim_ / 2;
  std::vector<float> inv_freq(freq_size);
  for (int i = 0; i < freq_size; i++) {
    float exponent = (float)(i * 2) / (float)qk_rope_head_dim_;
    inv_freq[i] = 1.0f / std::pow(rope_base, exponent);
  }
  rope_inv_freq_tensor_->SetShape(Shape{freq_size});
  rope_inv_freq_tensor_->CopyDataFrom(inv_freq.data(),
                                       sizeof(float) * freq_size,
                                       DeviceType::CPU, ctx_);
}

AsStatus MLAAttnOpCUDA::deviceInit() {
  int device_id;
  AS_CHECK_CUDA(cudaGetDevice(&device_id));
  AS_CHECK_CUDA(cudaGetDeviceProperties(&dprop_, device_id));

  DeviceType dev = ctx_->GetDeviceType();

  cuda::flashmla_clear_param(flash_mla_params_);
#ifdef FLASH_ATTN_V2
  cuda::flashv2_clear_param(flash_v2_params_);
#endif

  // MLA projection workspace tensors
  q_compressed_tensor_ = std::make_unique<AsTensor>(
      "mla_q_compressed", dev, dtype_, DataMode::DENSE, Shape{1, q_lora_rank_});
  q_normed_tensor_ = std::make_unique<AsTensor>(
      "mla_q_normed", dev, dtype_, DataMode::DENSE, Shape{1, q_lora_rank_});
  q_full_tensor_ = std::make_unique<AsTensor>(
      "mla_q_full", dev, dtype_, DataMode::DENSE,
      Shape{1, num_heads_ * qk_head_dim()});
  kv_compressed_tensor_ = std::make_unique<AsTensor>(
      "mla_kv_compressed", dev, dtype_, DataMode::DENSE,
      Shape{1, kv_cache_dim()});
  kv_normed_tensor_ = std::make_unique<AsTensor>(
      "mla_kv_normed", dev, dtype_, DataMode::DENSE, Shape{1, kv_lora_rank_});
  kv_full_tensor_ = std::make_unique<AsTensor>(
      "mla_kv_full", dev, dtype_, DataMode::DENSE,
      Shape{1, num_heads_ * (qk_nope_head_dim_ + v_head_dim_)});
  attn_output_tensor_ = std::make_unique<AsTensor>(
      "mla_attn_output", dev, dtype_, DataMode::DENSE,
      Shape{1, num_heads_ * v_head_dim_});

  // Assembled K/V for prefill
  k_assembled_tensor_ = std::make_unique<AsTensor>(
      "mla_k_assembled", dev, dtype_, DataMode::DENSE,
      Shape{1, num_heads_ * qk_head_dim()});
  v_assembled_tensor_ = std::make_unique<AsTensor>(
      "mla_v_assembled", dev, dtype_, DataMode::DENSE,
      Shape{1, num_heads_ * v_head_dim_});
  // Padded V and flash output (flash-attn requires K/V same head dim)
  v_padded_tensor_ = std::make_unique<AsTensor>(
      "mla_v_padded", dev, dtype_, DataMode::DENSE,
      Shape{1, num_heads_ * qk_head_dim()});
  flash_output_tensor_ = std::make_unique<AsTensor>(
      "mla_flash_output", dev, dtype_, DataMode::DENSE,
      Shape{1, num_heads_ * qk_head_dim()});

  // Decoder tensors
  decoder_q_tensor_ = std::make_unique<AsTensor>(
      "mla_decoder_q", dev, dtype_, DataMode::DENSE,
      Shape{1, 1, num_heads_, qk_head_dim()});
  decoder_seq_len_tensor_device_ = std::make_unique<AsTensor>(
      "mla_decoder_seq_len_dev", dev, DataType::INT32, DataMode::DENSE,
      Shape{1});
  decoder_seq_len_tensor_host_ = std::make_unique<AsTensor>(
      "mla_decoder_seq_len_host", DeviceType::CPU, DataType::INT32,
      DataMode::DENSE, Shape{1});

  // FlashMLA workspace
  splitkv_out_tensor_ = std::make_unique<AsTensor>(
      "mla_splitkv_out", dev, dtype_, DataMode::DENSE, Shape{1});
  splitkv_lse_tensor_ = std::make_unique<AsTensor>(
      "mla_splitkv_lse", dev, DataType::FLOAT32, DataMode::DENSE, Shape{1});
  tile_scheduler_metadata_tensor_ = std::make_unique<AsTensor>(
      "mla_tile_sched_meta", dev, DataType::INT8, DataMode::DENSE, Shape{1});
  num_splits_tensor_ = std::make_unique<AsTensor>(
      "mla_num_splits", dev, DataType::INT32, DataMode::DENSE, Shape{1});
  block_table_tensor_ = std::make_unique<AsTensor>(
      "mla_block_table", dev, DataType::INT32, DataMode::DENSE, Shape{1, 1});
  cache_seqlens_tensor_ = std::make_unique<AsTensor>(
      "mla_cache_seqlens", dev, DataType::INT32, DataMode::DENSE, Shape{1});

  // RoPE
  rope_inv_freq_tensor_ = std::make_unique<AsTensor>(
      "mla_rope_inv_freq", dev, DataType::FLOAT32, DataMode::DENSE,
      Shape{qk_rope_head_dim_ / 2});
  step_list_tensor_ = std::make_unique<AsTensor>(
      "mla_step_list", dev, DataType::INT32, DataMode::DENSE, Shape{1});

  // Prefill flash-attention workspace
  prefill_workspace_tensor_ = std::make_unique<AsTensor>(
      "mla_prefill_ws", dev, DataType::FLOAT32, DataMode::DENSE, Shape{1});

  // Span pointer arrays for non-contiguous paged cache
  kv_span_array_tensor_host_ = std::make_unique<AsTensor>(
      "mla_kv_span_array_host", DeviceType::CPU, DataType::POINTER,
      DataMode::DENSE, Shape{1});
  kv_span_array_tensor_device_ = std::make_unique<AsTensor>(
      "mla_kv_span_array_dev", dev, DataType::POINTER, DataMode::DENSE,
      Shape{1});

  computeInvFreq(rope_base_);

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus MLAAttnOpCUDA::deviceReshape(const RuntimeContext* runtime_ctx) {
  const int max_batch = ctx_->GetModelMaxBatch();
  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  if (runtime_ctx->is_context) {
    int M = batch_size_ * seq_len_;
    q_compressed_tensor_->SetShape(Shape{M, q_lora_rank_});
    q_normed_tensor_->SetShape(Shape{M, q_lora_rank_});
    q_full_tensor_->SetShape(Shape{M, num_heads_ * qk_head_dim()});
    kv_compressed_tensor_->SetShape(Shape{M, kv_cache_dim()});
    kv_normed_tensor_->SetShape(Shape{M, kv_lora_rank_});
    kv_full_tensor_->SetShape(
        Shape{M, num_heads_ * (qk_nope_head_dim_ + v_head_dim_)});
    attn_output_tensor_->SetShape(Shape{M, num_heads_ * v_head_dim_});
    k_assembled_tensor_->SetShape(Shape{M, num_heads_ * qk_head_dim()});
    v_assembled_tensor_->SetShape(Shape{M, num_heads_ * v_head_dim_});
    v_padded_tensor_->SetShape(Shape{M, num_heads_ * qk_head_dim()});
    flash_output_tensor_->SetShape(Shape{M, num_heads_ * qk_head_dim()});
    kv_span_array_tensor_host_->SetShape(Shape{1 * max_num_spans});
    kv_span_array_tensor_device_->SetShape(Shape{1 * max_num_spans});

    size_t lse_size = (size_t)batch_size_ * num_heads_ * seq_len_;
    prefill_workspace_tensor_->SetShape(Shape{(int64_t)(lse_size * 10)});
  } else {
    int M = max_batch;
    q_compressed_tensor_->SetShape(Shape{M, q_lora_rank_});
    q_normed_tensor_->SetShape(Shape{M, q_lora_rank_});
    q_full_tensor_->SetShape(Shape{M, num_heads_ * qk_head_dim()});
    kv_compressed_tensor_->SetShape(Shape{M, kv_cache_dim()});
    kv_normed_tensor_->SetShape(Shape{M, kv_lora_rank_});
    kv_full_tensor_->SetShape(
        Shape{M, num_heads_ * (qk_nope_head_dim_ + v_head_dim_)});
    attn_output_tensor_->SetShape(Shape{M, num_heads_ * v_head_dim_});

    decoder_q_tensor_->SetShape(
        Shape{max_batch, 1, num_heads_, qk_head_dim()});
    decoder_seq_len_tensor_device_->SetShape(Shape{max_batch});
    decoder_seq_len_tensor_host_->SetShape(Shape{max_batch});
    step_list_tensor_->SetShape(Shape{max_batch});

    block_table_tensor_->SetShape(Shape{max_batch, max_num_spans});
    cache_seqlens_tensor_->SetShape(Shape{max_batch});
    kv_span_array_tensor_host_->SetShape(Shape{max_batch * max_num_spans});
    kv_span_array_tensor_device_->SetShape(Shape{max_batch * max_num_spans});

    // FlashMLA decode workspace
    size_t splitkv_out_bytes, splitkv_lse_bytes, metadata_bytes,
        num_splits_bytes;
    cuda::flashmla_get_workspace_sizes(
        max_batch, num_heads_, kv_lora_rank_, max_seq_len, span_len,
        splitkv_out_bytes, splitkv_lse_bytes, metadata_bytes, num_splits_bytes);
    int ds = SizeofType(dtype_);
    splitkv_out_tensor_->SetShape(
        Shape{(int64_t)((splitkv_out_bytes + ds - 1) / ds)});
    splitkv_lse_tensor_->SetShape(
        Shape{(int64_t)((splitkv_lse_bytes + 3) / 4)});
    tile_scheduler_metadata_tensor_->SetShape(Shape{(int64_t)metadata_bytes});
    num_splits_tensor_->SetShape(
        Shape{(int64_t)((num_splits_bytes + 3) / 4)});
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus MLAAttnOpCUDA::runContext(RuntimeContext* runtime_ctx) {
  // MLA Prefill path - complete pipeline:
  //  1. q_a_proj -> q_a_norm -> q_b_proj -> decoupled RoPE on Q
  //  2. kv_a_proj -> strided RoPE on k_rope -> kv_a_norm -> kv_b_proj
  //  3. Assemble K (k_nope + k_rope broadcast) and V from kv_full
  //  4. Flash-attention(Q, K, V)
  //  5. Copy kv_compressed to span cache

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t stream = gpu_ctx->GetStream();

  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

  float eps = 1e-6f;
  int M = batch_size_ * seq_len_;
  const int span_len = ctx_->GetCacheSpanSize();

  auto functor = [&]<typename T>() {
    T* input = (T*)in_tensor->GetDataPtr();
    T* output = (T*)out_tensor->GetDataPtr();

    // Weight layout depends on q_lora_rank:
    //   V3 (q_lora_rank>0): [q_a_proj, q_a_norm, q_b_proj, kv_a_proj, kv_a_norm, kv_b_proj]
    //   V2 (q_lora_rank=0): [q_proj, kv_a_proj, kv_a_norm, kv_b_proj]
    int kv_weight_offset = (q_lora_rank_ > 0) ? 3 : 1;
    T* w_kv_a_proj = (T*)weights_[kv_weight_offset + 0]->GetDataPtr();
    T* w_kv_a_norm = (T*)weights_[kv_weight_offset + 1]->GetDataPtr();
    T* w_kv_b_proj = (T*)weights_[kv_weight_offset + 2]->GetDataPtr();

    T* q_full = (T*)q_full_tensor_->GetDataPtr();
    T* kv_compressed = (T*)kv_compressed_tensor_->GetDataPtr();
    T* kv_normed = (T*)kv_normed_tensor_->GetDataPtr();
    T* kv_full = (T*)kv_full_tensor_->GetDataPtr();
    T* k_assembled = (T*)k_assembled_tensor_->GetDataPtr();
    T* v_assembled = (T*)v_assembled_tensor_->GetDataPtr();
    float* inv_freq = (float*)rope_inv_freq_tensor_->GetDataPtr();

    // ---- Q projection ----
    int q_full_dim = num_heads_ * qk_head_dim();
    if (q_lora_rank_ > 0) {
      // V3: q_a_proj -> q_a_norm -> q_b_proj
      T* w_q_a_proj = (T*)weights_[0]->GetDataPtr();
      T* w_q_a_norm = (T*)weights_[1]->GetDataPtr();
      T* w_q_b_proj = (T*)weights_[2]->GetDataPtr();
      T* q_compressed = (T*)q_compressed_tensor_->GetDataPtr();
      T* q_normed = (T*)q_normed_tensor_->GetDataPtr();

      cuda::GemmWraper<T>(q_compressed, input, w_q_a_proj, nullptr,
                          M, q_lora_rank_, hidden_size_, false, false,
                          hidden_size_, q_lora_rank_, q_lora_rank_,
                          1.0f, 0.0f, nullptr, cublas_handle, stream);

      cuda::LayerNormNoBetaKernelLauncher<T>(
          q_normed, q_compressed, nullptr, w_q_a_norm,
          M, q_lora_rank_, eps, stream);

      cuda::GemmWraper<T>(q_full, q_normed, w_q_b_proj, nullptr,
                          M, q_full_dim, q_lora_rank_, false, false,
                          q_lora_rank_, q_full_dim, q_full_dim,
                          1.0f, 0.0f, nullptr, cublas_handle, stream);
    } else {
      // V2: direct q_proj
      T* w_q_proj = (T*)weights_[0]->GetDataPtr();
      cuda::GemmWraper<T>(q_full, input, w_q_proj, nullptr,
                          M, q_full_dim, hidden_size_, false, false,
                          hidden_size_, q_full_dim, q_full_dim,
                          1.0f, 0.0f, nullptr, cublas_handle, stream);
    }

    // Apply decoupled RoPE to Q (last d_rope dims of each head)
    cuda::MLARoPEQLauncher<T>(
        q_full, inv_freq, nullptr, nullptr,
        M, num_heads_, qk_nope_head_dim_, qk_rope_head_dim_, seq_len_,
        stream);

    // ---- KV projection chain ----
    int kv_compressed_dim = kv_cache_dim();
    cuda::GemmWraper<T>(kv_compressed, input, w_kv_a_proj, nullptr,
                        M, kv_compressed_dim, hidden_size_, false, false,
                        hidden_size_, kv_compressed_dim, kv_compressed_dim,
                        1.0f, 0.0f, nullptr, cublas_handle, stream);

    // Apply strided RoPE to k_rope within kv_compressed (in-place)
    cuda::MLAStridedRoPEKLauncher<T>(
        kv_compressed, inv_freq, nullptr, M, qk_rope_head_dim_,
        kv_lora_rank_, stream);

    // RMSNorm on latent part only (first kv_lora_rank dims).
    // kv_compressed has stride kv_compressed_dim per token, but LayerNorm
    // expects contiguous input. Copy latent part to kv_normed first.
    AS_CHECK_CUDA(cudaMemcpy2DAsync(
        kv_normed, kv_lora_rank_ * sizeof(T),
        kv_compressed, kv_compressed_dim * sizeof(T),
        kv_lora_rank_ * sizeof(T), M,
        cudaMemcpyDeviceToDevice, stream));
    cuda::LayerNormNoBetaKernelLauncher<T>(
        kv_normed, kv_normed, nullptr, w_kv_a_norm,
        M, kv_lora_rank_, eps, stream);

    // KV up-projection
    int kv_full_dim = num_heads_ * (qk_nope_head_dim_ + v_head_dim_);
    cuda::GemmWraper<T>(kv_full, kv_normed, w_kv_b_proj, nullptr,
                        M, kv_full_dim, kv_lora_rank_, false, false,
                        kv_lora_rank_, kv_full_dim, kv_full_dim,
                        1.0f, 0.0f, nullptr, cublas_handle, stream);

    // ---- Assemble K and V ----
    // k_rope for assembly comes from kv_compressed[:, kv_lora_rank_:]
    // It's strided with stride = kv_compressed_dim
    T* k_rope_ptr = kv_compressed + kv_lora_rank_;
    cuda::MLAKVAssembleLauncher<T>(
        k_assembled, v_assembled, kv_full, k_rope_ptr,
        M, num_heads_, qk_nope_head_dim_, qk_rope_head_dim_, v_head_dim_,
        kv_compressed_dim,  // k_rope stride
        stream);

    // ---- Copy normed latent back to kv_compressed for consistent cache format ----
    // Cache stores [normed_latent, roped_k_rope] so decode kernel can use
    // absorbed attention without per-token RMSNorm.
    AS_CHECK_CUDA(cudaMemcpy2DAsync(
        kv_compressed, kv_compressed_dim * sizeof(T),
        kv_normed, kv_lora_rank_ * sizeof(T),
        kv_lora_rank_ * sizeof(T), M,
        cudaMemcpyDeviceToDevice, stream));

    // ---- Store compressed KV to span cache ----
    GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
    const AsTensor& kv_cache_ptrs =
        gen_ctx->virtual_k_cache->GetCache(layer_num_, 0);

    // Build span pointer array for the copy kernel
    auto& layer_cache = gen_ctx->virtual_k_cache->GetLayerCache();
    if (layer_num_ < (int)layer_cache.size() && layer_cache[layer_num_]) {
      auto& cache_vec = layer_cache[layer_num_]->GetCacheVector();
      int num_spans = (int)cache_vec.size();
      kv_span_array_tensor_host_->SetShape(Shape{num_spans});
      kv_span_array_tensor_device_->SetShape(Shape{num_spans});
      void** host_ptrs = (void**)kv_span_array_tensor_host_->GetDataPtr();
      for (int s = 0; s < num_spans; s++) {
        host_ptrs[s] = cache_vec[s]->Data();
      }
      AS_CHECK_CUDA(cudaMemcpyAsync(
          kv_span_array_tensor_device_->GetDataPtr(), host_ptrs,
          num_spans * sizeof(void*), cudaMemcpyHostToDevice, stream));

      // Copy kv_compressed to span cache
      cuda::MLACopyToSpanCacheLauncher<T>(
          (T* const*)kv_span_array_tensor_device_->GetDataPtr(),
          kv_compressed, M, kv_compressed_dim, span_len,
          gen_ctx->step,  // start_pos
          stream);
    }

    // ---- Flash-attention prefill ----
#ifdef FLASH_ATTN_V2
    cudaDataType_t cuda_dtype =
        (dtype_ == DataType::BFLOAT16) ? CUDA_R_16BF : CUDA_R_16F;

    // FlashAttn V2 requires K and V to have the same head dim.
    // MLA has K head_dim=192 but V head_dim=128.
    // Pad V to qk_head_dim (192) with zeros, run flash-attn into a
    // temporary buffer, then extract the first v_head_dim elements per head.
    int qk_hd = qk_head_dim();
    T* v_padded = (T*)v_padded_tensor_->GetDataPtr();
    T* flash_out = (T*)flash_output_tensor_->GetDataPtr();

    // Zero the padded buffer, then copy v_assembled (strided)
    AS_CHECK_CUDA(cudaMemsetAsync(
        v_padded, 0, (size_t)M * num_heads_ * qk_hd * sizeof(T), stream));
    AS_CHECK_CUDA(cudaMemcpy2DAsync(
        v_padded, qk_hd * sizeof(T),
        v_assembled, v_head_dim_ * sizeof(T),
        v_head_dim_ * sizeof(T), (size_t)M * num_heads_,
        cudaMemcpyDeviceToDevice, stream));

    cuda::flashv2_set_static_param(
        flash_v2_params_, dprop_, cuda_dtype,
        batch_size_, seq_len_, seq_len_,
        num_heads_, num_heads_,
        qk_hd,
        cuda::FlashQKVFormat::CONTINUOUS,
        true);

    float softmax_scale = 1.0f / std::sqrt((float)qk_hd);

    cuda::flashv2_set_runtime_param(
        flash_v2_params_, q_full, k_assembled, v_padded,
        flash_out, prefill_workspace_tensor_->GetDataPtr(), softmax_scale);

    cuda::flashv2_dispatch(flash_v2_params_, stream);

    // Extract first v_head_dim elements per head from flash output
    AS_CHECK_CUDA(cudaMemcpy2DAsync(
        output, v_head_dim_ * sizeof(T),
        flash_out, qk_hd * sizeof(T),
        v_head_dim_ * sizeof(T), (size_t)M * num_heads_,
        cudaMemcpyDeviceToDevice, stream));
#else
    // Fallback: zero output if flash-attention not available
    AS_CHECK_CUDA(cudaMemsetAsync(
        output, 0, (size_t)M * num_heads_ * v_head_dim_ * sizeof(T), stream));
#endif
  };
  DispatchCUDA(dtype_, functor);

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus MLAAttnOpCUDA::runDecoder(RuntimeContext* runtime_ctx) {
  // MLA Decode path - complete pipeline:
  //  1. Q projection chain with position-aware RoPE
  //  2. KV compression + strided RoPE + append to span cache
  //  3. FlashMLA decode attention (or fallback span-attention)

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t stream = gpu_ctx->GetStream();

  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

  const int span_len = ctx_->GetCacheSpanSize();
  const int max_seq_len = ctx_->GetModelMaxLength();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  float eps = 1e-6f;
  int M = batch_size_;

  auto functor = [&]<typename T>() {
    T* input = (T*)in_tensor->GetDataPtr();
    T* output = (T*)out_tensor->GetDataPtr();

    int kv_weight_offset = (q_lora_rank_ > 0) ? 3 : 1;
    T* w_kv_a_proj = (T*)weights_[kv_weight_offset + 0]->GetDataPtr();
    T* w_kv_a_norm = (T*)weights_[kv_weight_offset + 1]->GetDataPtr();

    T* q_full = (T*)q_full_tensor_->GetDataPtr();
    T* kv_compressed = (T*)kv_compressed_tensor_->GetDataPtr();
    T* kv_normed = (T*)kv_normed_tensor_->GetDataPtr();
    float* inv_freq = (float*)rope_inv_freq_tensor_->GetDataPtr();

    // ---- Q projection ----
    int q_full_dim = num_heads_ * qk_head_dim();
    if (q_lora_rank_ > 0) {
      T* w_q_a_proj = (T*)weights_[0]->GetDataPtr();
      T* w_q_a_norm = (T*)weights_[1]->GetDataPtr();
      T* w_q_b_proj = (T*)weights_[2]->GetDataPtr();
      T* q_compressed = (T*)q_compressed_tensor_->GetDataPtr();
      T* q_normed = (T*)q_normed_tensor_->GetDataPtr();

      cuda::GemmWraper<T>(q_compressed, input, w_q_a_proj, nullptr,
                          M, q_lora_rank_, hidden_size_, false, false,
                          hidden_size_, q_lora_rank_, q_lora_rank_,
                          1.0f, 0.0f, nullptr, cublas_handle, stream);

      cuda::LayerNormNoBetaKernelLauncher<T>(
          q_normed, q_compressed, nullptr, w_q_a_norm,
          M, q_lora_rank_, eps, stream);

      cuda::GemmWraper<T>(q_full, q_normed, w_q_b_proj, nullptr,
                          M, q_full_dim, q_lora_rank_, false, false,
                          q_lora_rank_, q_full_dim, q_full_dim,
                          1.0f, 0.0f, nullptr, cublas_handle, stream);
    } else {
      T* w_q_proj = (T*)weights_[0]->GetDataPtr();
      cuda::GemmWraper<T>(q_full, input, w_q_proj, nullptr,
                          M, q_full_dim, hidden_size_, false, false,
                          hidden_size_, q_full_dim, q_full_dim,
                          1.0f, 0.0f, nullptr, cublas_handle, stream);
    }

    // Build step list for position-aware RoPE
    std::vector<int> host_steps(batch_size_);
    for (int b = 0; b < batch_size_; b++) {
      host_steps[b] = runtime_ctx->GetGenCtx(b)->step;
    }
    step_list_tensor_->SetShape(Shape{batch_size_});
    AS_CHECK_CUDA(cudaMemcpyAsync(
        step_list_tensor_->GetDataPtr(), host_steps.data(),
        batch_size_ * sizeof(int), cudaMemcpyHostToDevice, stream));
    int* step_dev = (int*)step_list_tensor_->GetDataPtr();

    cuda::MLARoPEQLauncher<T>(
        q_full, inv_freq, step_dev, nullptr,
        M, num_heads_, qk_nope_head_dim_, qk_rope_head_dim_, 1, stream);

    // ---- KV compression ----
    int kv_compressed_dim = kv_cache_dim();
    cuda::GemmWraper<T>(kv_compressed, input, w_kv_a_proj, nullptr,
                        M, kv_compressed_dim, hidden_size_, false, false,
                        hidden_size_, kv_compressed_dim, kv_compressed_dim,
                        1.0f, 0.0f, nullptr, cublas_handle, stream);

    // Strided RoPE on k_rope portion
    cuda::MLAStridedRoPEKLauncher<T>(
        kv_compressed, inv_freq, step_dev, M, qk_rope_head_dim_,
        kv_lora_rank_, stream);

    // RMSNorm on latent part, then copy back for cache
    cuda::LayerNormNoBetaKernelLauncher<T>(
        kv_normed, kv_compressed, nullptr, w_kv_a_norm,
        M, kv_lora_rank_, eps, stream);
    AS_CHECK_CUDA(cudaMemcpyAsync(
        kv_compressed, kv_normed, (size_t)M * kv_lora_rank_ * sizeof(T),
        cudaMemcpyDeviceToDevice, stream));

    // ---- Append to span cache ----
    for (int b = 0; b < batch_size_; b++) {
      GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(b);
      int step = gen_ctx->step;
      auto& layer_cache = gen_ctx->virtual_k_cache->GetLayerCache();
      if (layer_num_ < (int)layer_cache.size() && layer_cache[layer_num_]) {
        auto& cache_vec = layer_cache[layer_num_]->GetCacheVector();
        int span_idx = step / span_len;
        int span_offset = step % span_len;
        if (span_idx < (int)cache_vec.size()) {
          T* dst = (T*)cache_vec[span_idx]->Data() +
                   (int64_t)span_offset * kv_compressed_dim;
          T* src = kv_compressed + b * kv_compressed_dim;
          AS_CHECK_CUDA(cudaMemcpyAsync(
              dst, src, kv_compressed_dim * sizeof(T),
              cudaMemcpyDeviceToDevice, stream));
        }
      }
    }

    // ---- Build span pointer arrays for decode attention ----
    std::vector<int> host_cache_seqlens(batch_size_);
    for (int b = 0; b < batch_size_; b++) {
      GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(b);
      host_cache_seqlens[b] = gen_ctx->step + 1;
    }

    cache_seqlens_tensor_->SetShape(Shape{batch_size_});
    AS_CHECK_CUDA(cudaMemcpyAsync(
        cache_seqlens_tensor_->GetDataPtr(), host_cache_seqlens.data(),
        batch_size_ * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Gather span data pointers for each batch's cache
    int total_span_ptrs = batch_size_ * max_num_spans;
    kv_span_array_tensor_host_->SetShape(Shape{total_span_ptrs});
    kv_span_array_tensor_device_->SetShape(Shape{total_span_ptrs});
    void** host_span_ptrs =
        (void**)kv_span_array_tensor_host_->GetDataPtr();
    memset(host_span_ptrs, 0, total_span_ptrs * sizeof(void*));

    for (int b = 0; b < batch_size_; b++) {
      GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(b);
      auto& layer_cache = gen_ctx->virtual_k_cache->GetLayerCache();
      if (layer_num_ < (int)layer_cache.size() && layer_cache[layer_num_]) {
        auto& cache_vec = layer_cache[layer_num_]->GetCacheVector();
        for (int s = 0; s < (int)cache_vec.size() && s < max_num_spans; s++) {
          host_span_ptrs[b * max_num_spans + s] = cache_vec[s]->Data();
        }
      }
    }

    AS_CHECK_CUDA(cudaMemcpyAsync(
        kv_span_array_tensor_device_->GetDataPtr(), host_span_ptrs,
        total_span_ptrs * sizeof(void*), cudaMemcpyHostToDevice, stream));

    // ---- MLA decode attention ----
    float softmax_scale = 1.0f / std::sqrt((float)qk_head_dim());

    // Use naive absorbed-attention kernel (fallback for when FlashMLA lib
    // is not linked).  The kernel operates directly on the paged compressed
    // KV cache:  q_absorbed @ kv_latent + q_rope @ k_rope_cached.
    int kv_b_offset = (q_lora_rank_ > 0) ? 5 : 3;
    T* w_kv_b = (T*)weights_[kv_b_offset]->GetDataPtr();
    cuda::MLADecodeNaiveLauncher<T>(
        output, q_full, w_kv_b,
        (void* const*)kv_span_array_tensor_device_->GetDataPtr(),
        (int*)cache_seqlens_tensor_->GetDataPtr(),
        batch_size_, num_heads_, kv_lora_rank_,
        qk_nope_head_dim_, qk_rope_head_dim_, v_head_dim_,
        kv_cache_dim(), span_len, max_num_spans, softmax_scale, stream);
  };
  DispatchCUDA(dtype_, functor);

  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(MLAAttention, CUDA, MLAAttnOpCUDA)
REGISTER_OP(DecOptMLA, CUDA, MLAAttnOpCUDA)

}  // namespace allspark
#endif  // ENABLE_CUDA
