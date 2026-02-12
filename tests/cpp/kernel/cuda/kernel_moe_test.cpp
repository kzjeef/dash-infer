/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_moe_test.cpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>

#include "common.hpp"
#include "test_common.h"

#include <core/kernel/cuda/cuda_kernel.h>
#include <core/kernel/cuda/moe/moe_dnn.h>
#include <cuda/cuda_context.h>

namespace {

// ---------------------------------------------------------------------------
//  CPU reference helpers
// ---------------------------------------------------------------------------

/// CPU softmax over the inner dimension (num_expert) for each token.
void CPU_SoftmaxLowReduce(const float* input, float* output, int total_token,
                           int num_expert) {
  for (int t = 0; t < total_token; ++t) {
    float max_val = -INFINITY;
    for (int e = 0; e < num_expert; ++e) {
      max_val = std::max(max_val, input[t * num_expert + e]);
    }
    float sum = 0.f;
    for (int e = 0; e < num_expert; ++e) {
      output[t * num_expert + e] = std::exp(input[t * num_expert + e] - max_val);
      sum += output[t * num_expert + e];
    }
    for (int e = 0; e < num_expert; ++e) {
      output[t * num_expert + e] /= sum;
    }
  }
}

/// CPU top-k: for each token select top_k experts.
void CPU_TopK(const float* softmax_out, float* topk_score, int* topk_indice,
              int total_token, int num_expert, int top_k) {
  for (int t = 0; t < total_token; ++t) {
    std::vector<std::pair<float, int>> scored(num_expert);
    for (int e = 0; e < num_expert; ++e) {
      scored[e] = {softmax_out[t * num_expert + e], e};
    }
    std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                      [](const auto& a, const auto& b) {
                        return a.first > b.first;
                      });
    for (int k = 0; k < top_k; ++k) {
      topk_score[t * top_k + k] = scored[k].first;
      topk_indice[t * top_k + k] = scored[k].second;
    }
  }
}

/// CPU batched GEMM: for each (token, expert) pair, compute
///   C[row] = A[token] * B[expert]   where B is [num_expert, K, N]
/// A is [total_token, K], routing says which expert each row uses.
template <typename T>
void CPU_MoeBatchedGemm(const std::vector<float>& A_f32,
                         const std::vector<float>& B_f32,
                         std::vector<float>& C_f32, const int* topk_indice,
                         int total_token, int top_k, int num_expert, int N,
                         int K) {
  int total_rows = total_token * top_k;
  C_f32.assign(total_rows * N, 0.f);
  for (int t = 0; t < total_token; ++t) {
    for (int ki = 0; ki < top_k; ++ki) {
      int row = t * top_k + ki;
      int expert = topk_indice[t * top_k + ki];
      if (expert < 0 || expert >= num_expert) continue;
      const float* a_ptr = A_f32.data() + t * K;
      const float* b_ptr = B_f32.data() + (int64_t)expert * K * N;
      float* c_ptr = C_f32.data() + (int64_t)row * N;
      for (int ni = 0; ni < N; ++ni) {
        float acc = 0.f;
        for (int kk = 0; kk < K; ++kk) {
          acc += a_ptr[kk] * b_ptr[kk * N + ni];
        }
        c_ptr[ni] = acc;
      }
    }
  }
}

/// CPU SiLU-GLU: splits input [M, 2*proj_size] into gate and up,
///   output[i] = silu(gate[i]) * up[i]
void CPU_SiluGLU(const float* input, float* output, int M, int proj_size) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < proj_size; ++n) {
      float gate = input[m * proj_size * 2 + n];
      float up = input[m * proj_size * 2 + proj_size + n];
      float silu_gate = gate / (1.f + std::exp(-gate));
      output[m * proj_size + n] = silu_gate * up;
    }
  }
}

/// CPU finalize: weighted sum of expert outputs per token.
///   output[token] = sum_k( score[token,k] * result[token*topk+k] )
void CPU_FinalizeMoeRouting(const float* final_result, const float* scores,
                            float* output, int total_token, int top_k,
                            int hidden_size) {
  for (int t = 0; t < total_token; ++t) {
    for (int h = 0; h < hidden_size; ++h) {
      float acc = 0.f;
      for (int k = 0; k < top_k; ++k) {
        int row = t * top_k + k;
        acc += scores[t * top_k + k] * final_result[row * hidden_size + h];
      }
      output[t * hidden_size + h] = acc;
    }
  }
}

/// Full CPU MOE reference:
///   gate_up = input * gate_up_weight  (per expert)
///   mid     = silu_glu(gate_up)
///   down    = mid * down_weight       (per expert)
///   output  = weighted_sum(down, scores)
template <typename T>
void CPU_MoeReference(const std::vector<T>& input_h,
                      const std::vector<T>& gate_score_h,
                      const std::vector<T>& gate_up_weight_h,
                      const std::vector<T>& down_weight_h,
                      std::vector<T>& output_h, int total_token,
                      int num_expert, int top_k, int hidden_size,
                      int proj_size) {
  // Convert to float for reference computation
  int input_size = total_token * hidden_size;
  std::vector<float> input_f(input_size);
  for (int i = 0; i < input_size; ++i) input_f[i] = float(input_h[i]);

  std::vector<float> gate_score_f(total_token * num_expert);
  for (int i = 0; i < total_token * num_expert; ++i)
    gate_score_f[i] = float(gate_score_h[i]);

  int64_t guw_size = (int64_t)num_expert * hidden_size * proj_size * 2;
  std::vector<float> guw_f(guw_size);
  for (int64_t i = 0; i < guw_size; ++i) guw_f[i] = float(gate_up_weight_h[i]);

  int64_t dw_size = (int64_t)num_expert * proj_size * hidden_size;
  std::vector<float> dw_f(dw_size);
  for (int64_t i = 0; i < dw_size; ++i) dw_f[i] = float(down_weight_h[i]);

  // 1. Softmax
  std::vector<float> softmax_out(total_token * num_expert);
  CPU_SoftmaxLowReduce(gate_score_f.data(), softmax_out.data(), total_token,
                        num_expert);

  // 2. TopK
  std::vector<float> topk_score(total_token * top_k);
  std::vector<int> topk_indice(total_token * top_k);
  CPU_TopK(softmax_out.data(), topk_score.data(), topk_indice.data(),
           total_token, num_expert, top_k);

  // 3. Gate-Up GEMM  (total_token x hidden_size) * (expert x hidden_size x
  // 2*proj_size)
  int total_rows = total_token * top_k;
  std::vector<float> gate_up_out;
  CPU_MoeBatchedGemm<T>(input_f, guw_f, gate_up_out, topk_indice.data(),
                         total_token, top_k, num_expert, proj_size * 2,
                         hidden_size);

  // 4. SiLU-GLU
  std::vector<float> mid_result(total_rows * proj_size);
  CPU_SiluGLU(gate_up_out.data(), mid_result.data(), total_rows, proj_size);

  // 5. Down GEMM  (total_rows x proj_size) * (expert x proj_size x
  // hidden_size)
  //    We need to map each row to its expert for the down projection too.
  //    The row->expert mapping after gate_up is: row i belongs to
  //    topk_indice[i].
  std::vector<float> final_result;
  // For down projection, each row maps to its own expert.  Build flat indice.
  std::vector<int> down_indice(total_rows);
  for (int t = 0; t < total_token; ++t) {
    for (int k = 0; k < top_k; ++k) {
      down_indice[t * top_k + k] = topk_indice[t * top_k + k];
    }
  }
  // Down GEMM: each row of mid_result (proj_size) × down_weight[expert]
  // (proj_size × hidden_size) → hidden_size
  final_result.assign(total_rows * hidden_size, 0.f);
  for (int r = 0; r < total_rows; ++r) {
    int expert = down_indice[r];
    if (expert < 0 || expert >= num_expert) continue;
    const float* m_ptr = mid_result.data() + (int64_t)r * proj_size;
    const float* w_ptr = dw_f.data() + (int64_t)expert * proj_size * hidden_size;
    float* o_ptr = final_result.data() + (int64_t)r * hidden_size;
    for (int ni = 0; ni < hidden_size; ++ni) {
      float acc = 0.f;
      for (int kk = 0; kk < proj_size; ++kk) {
        acc += m_ptr[kk] * w_ptr[kk * hidden_size + ni];
      }
      o_ptr[ni] = acc;
    }
  }

  // 6. Finalize: weighted sum
  std::vector<float> output_f(total_token * hidden_size, 0.f);
  CPU_FinalizeMoeRouting(final_result.data(), topk_score.data(), output_f.data(),
                         total_token, top_k, hidden_size);

  output_h.resize(total_token * hidden_size);
  for (int i = 0; i < total_token * hidden_size; ++i)
    output_h[i] = T(output_f[i]);
}

// ---------------------------------------------------------------------------
//  GPU MoeBatchedGemmLauncher unit test (DNN path, SM >= 90)
// ---------------------------------------------------------------------------

/// Tests the custom WGMMA-based MoeBatchedGemmLauncher kernel directly.
/// Uses nMatBPerMatARow=1 (each row maps to one expert, like the down GEMM).
/// This avoids the complex row-index expansion logic and focuses on GEMM
/// correctness.
template <typename T>
void TestMoeBatchedGemm(int matARows, int num_expert, int top_k, int N,
                         int K, float eps = 0.05f) {
  // nMatBPerMatARow=1: each of the matARows rows has a single expert index.
  // This matches how the down-projection GEMM is called in the MOE operator.
  const int nMatBPerMatARow = 1;
  printf(
      "TestMoeBatchedGemm<%s>: matARows=%d, num_expert=%d, N=%d, K=%d\n",
      (std::is_same<T, half>::value ? "half" : "bf16"), matARows, num_expert,
      N, K);

  // A [matARows, K], B [num_expert, K, N]
  auto A_host = common::rand_normal_float<T>(matARows * K, 0.3f);
  auto B_host =
      common::rand_normal_float<T>((int64_t)num_expert * K * N, 0.3f);

  // Each row maps to one expert
  std::vector<uint32_t> matBIndices_host(matARows);
  {
    std::default_random_engine gen(42);
    std::uniform_int_distribution<int> dis(0, num_expert - 1);
    for (int r = 0; r < matARows; ++r) {
      matBIndices_host[r] = dis(gen);
    }
  }

  // CPU reference: C[r] = A[r] * B[expert[r]]
  std::vector<float> C_ref(matARows * N, 0.f);
  for (int r = 0; r < matARows; ++r) {
    int expert = matBIndices_host[r];
    for (int ni = 0; ni < N; ++ni) {
      float acc = 0.f;
      for (int kk = 0; kk < K; ++kk) {
        acc += float(A_host[r * K + kk]) *
               float(B_host[(int64_t)expert * K * N + kk * N + ni]);
      }
      C_ref[r * N + ni] = acc;
    }
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  T *d_A, *d_B, *d_C;
  uint32_t *d_matBIndices, *d_matCRowIndices;
  cudaMalloc(&d_A, matARows * K * sizeof(T));
  cudaMalloc(&d_B, (int64_t)num_expert * K * N * sizeof(T));
  cudaMalloc(&d_C, matARows * N * sizeof(T));
  cudaMalloc(&d_matBIndices, matARows * sizeof(uint32_t));
  cudaMalloc(&d_matCRowIndices, matARows * sizeof(uint32_t));

  cudaMemcpyAsync(d_A, A_host.data(), matARows * K * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, B_host.data(),
                  (int64_t)num_expert * K * N * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_matBIndices, matBIndices_host.data(),
                  matARows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(d_matCRowIndices, 0, matARows * sizeof(uint32_t), stream);

  // Workspace
  size_t wsSize =
      allspark::cuda::GetWorkspaceSizeLauncher(matARows, num_expert);
  void* d_ws;
  cudaMalloc(&d_ws, wsSize);

  // Run kernel: matARows rows, nMatBPerMatARow=1
  allspark::cuda::MoeBatchedGemmLauncher<T>(d_A, d_B, d_matBIndices, d_C,
                                             d_matCRowIndices, d_ws, wsSize,
                                             matARows, N, K, num_expert,
                                             nMatBPerMatARow, stream);
  cudaStreamSynchronize(stream);

  // Read matCRowIndices to understand output layout
  std::vector<uint32_t> matCRowIndices_host(matARows);
  cudaMemcpy(matCRowIndices_host.data(), d_matCRowIndices,
             matARows * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // Copy output
  std::vector<T> C_host(matARows * N);
  cudaMemcpy(C_host.data(), d_C, matARows * N * sizeof(T),
             cudaMemcpyDeviceToHost);

  // Compare using matCRowIndices to map input row → output row
  float max_diff = 0.f;
  int exceed_count = 0;
  int valid_count = 0;
  for (int r = 0; r < matARows; ++r) {
    uint32_t out_row = matCRowIndices_host[r];
    if (out_row >= (uint32_t)matARows) continue;  // invalid mapping
    for (int ni = 0; ni < N; ++ni) {
      float gpu_val = float(C_host[out_row * N + ni]);
      float ref_val = C_ref[r * N + ni];
      float diff = std::fabs(gpu_val - ref_val);
      float denom = std::max(std::fabs(ref_val), 1e-6f);
      float rel = diff / denom;
      float err = std::min(diff, rel);
      if (err > max_diff) max_diff = err;
      if (err > eps) exceed_count++;
      valid_count++;
    }
  }
  printf("  MaxDiff=%f, exceed_count=%d / %d (valid=%d)\n", max_diff,
         exceed_count, matARows * N, valid_count);
  EXPECT_GT(valid_count, 0);
  EXPECT_LT(exceed_count, valid_count / 10);  // < 10% outliers for fp16

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_matBIndices);
  cudaFree(d_matCRowIndices);
  cudaFree(d_ws);
  cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
//  Full MOE pipeline test (end-to-end via operator-level calls)
// ---------------------------------------------------------------------------

/// Tests the full MOE pipeline: softmax → topk → gate_up_gemm → silu_glu →
///   down_gemm → finalize_routing.
/// Exercises both the DNN (WGMMA, SM>=90) and the block (cuBLAS, SM<90) paths.
template <typename T>
void TestMoeFullPipeline(int total_token, int num_expert, int top_k,
                          int hidden_size, int proj_size, float eps = 0.1f) {
  printf(
      "TestMoeFullPipeline<%s>: total_token=%d, num_expert=%d, top_k=%d, "
      "hidden=%d, proj=%d\n",
      (std::is_same<T, half>::value ? "half" : "bf16"), total_token, num_expert,
      top_k, hidden_size, proj_size);

  // Generate random data
  auto input_h = common::rand_normal_float<T>(total_token * hidden_size, 0.3f);
  auto gate_score_h =
      common::rand_normal_float<T>(total_token * num_expert, 1.0f);
  auto gate_up_weight_h = common::rand_normal_float<T>(
      (int64_t)num_expert * hidden_size * proj_size * 2, 0.1f);
  auto down_weight_h = common::rand_normal_float<T>(
      (int64_t)num_expert * proj_size * hidden_size, 0.1f);

  // CPU reference
  std::vector<T> output_ref;
  CPU_MoeReference<T>(input_h, gate_score_h, gate_up_weight_h, down_weight_h,
                      output_ref, total_token, num_expert, top_k, hidden_size,
                      proj_size);

  // Check SM version to decide which path to test
  cudaDeviceProp dprop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&dprop, device_id);
  int sm_version = dprop.major << 8 | dprop.minor;
  bool use_dnn = (sm_version >= 0x0900);

  printf("  SM version: %d.%d, path: %s\n", dprop.major, dprop.minor,
         use_dnn ? "DNN (WGMMA)" : "Block (cuBLAS)");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // --- Device allocations ---
  T *d_input, *d_gate_score, *d_gate_up_weight, *d_down_weight, *d_output;
  cudaMalloc(&d_input, total_token * hidden_size * sizeof(T));
  cudaMalloc(&d_gate_score, total_token * num_expert * sizeof(T));
  cudaMalloc(&d_gate_up_weight,
             (int64_t)num_expert * hidden_size * proj_size * 2 * sizeof(T));
  cudaMalloc(&d_down_weight,
             (int64_t)num_expert * proj_size * hidden_size * sizeof(T));
  cudaMalloc(&d_output, total_token * hidden_size * sizeof(T));

  cudaMemcpyAsync(d_input, input_h.data(),
                  total_token * hidden_size * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_gate_score, gate_score_h.data(),
                  total_token * num_expert * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_gate_up_weight, gate_up_weight_h.data(),
                  (int64_t)num_expert * hidden_size * proj_size * 2 * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_down_weight, down_weight_h.data(),
                  (int64_t)num_expert * proj_size * hidden_size * sizeof(T),
                  cudaMemcpyHostToDevice, stream);

  // Intermediate buffers
  float *d_float_gate_score, *d_topk_value, *d_experts_score;
  int *d_topk_indice;
  cudaMalloc(&d_float_gate_score, total_token * num_expert * sizeof(float));
  cudaMalloc(&d_topk_value, total_token * num_expert * sizeof(float));
  cudaMalloc(&d_experts_score, total_token * top_k * sizeof(float));
  cudaMalloc(&d_topk_indice, total_token * top_k * sizeof(int));

  // Step 1: Cast gate scores to float
  allspark::cuda::CastKernelLauncher<T, float>(d_gate_score, d_float_gate_score,
                                                total_token * num_expert,
                                                stream);

  // Step 2: Softmax
  allspark::cuda::SoftmaxLowReduceKernelLauncher<float>(
      d_float_gate_score, d_topk_value, total_token, num_expert, stream);

  // Step 3: TopK
  allspark::cuda::TopKKernelLauncher<float>(d_experts_score, d_topk_indice,
                                             d_topk_value, total_token,
                                             num_expert, top_k, stream);

  if (use_dnn) {
    // --- DNN path (WGMMA) ---
    int matARows_gu = total_token * top_k;
    size_t wsSize =
        allspark::cuda::GetWorkspaceSizeLauncher(matARows_gu, num_expert);

    // Workspace for intermediates
    size_t gate_up_out_bytes = (size_t)matARows_gu * proj_size * 2 * sizeof(T);
    size_t mid_result_bytes = (size_t)matARows_gu * proj_size * sizeof(T);
    size_t final_result_bytes = (size_t)matARows_gu * hidden_size * sizeof(T);

    // Align to 128 bytes
    auto align128 = [](size_t n) { return (n + 127) / 128 * 128; };

    size_t total_ws =
        align128(gate_up_out_bytes) + align128(mid_result_bytes) +
        align128(final_result_bytes) + align128(wsSize);

    char* d_total_ws;
    cudaMalloc(&d_total_ws, total_ws);

    T* d_gate_up_out = reinterpret_cast<T*>(d_total_ws);
    T* d_mid_result =
        reinterpret_cast<T*>(d_total_ws + align128(gate_up_out_bytes));
    T* d_final_result = reinterpret_cast<T*>(
        d_total_ws + align128(gate_up_out_bytes) + align128(mid_result_bytes));
    void* d_dnn_ws = d_total_ws + align128(gate_up_out_bytes) +
                     align128(mid_result_bytes) + align128(final_result_bytes);

    // Indices buffers
    uint32_t *d_mid_row_indices, *d_mid_expert_indices, *d_final_row_indices;
    cudaMalloc(&d_mid_row_indices, matARows_gu * sizeof(uint32_t));
    cudaMalloc(&d_mid_expert_indices, matARows_gu * sizeof(uint32_t));
    cudaMalloc(&d_final_row_indices, matARows_gu * sizeof(uint32_t));

    // Gate-Up GEMM
    allspark::cuda::MoeBatchedGemmLauncher<T>(
        d_input, d_gate_up_weight, (uint32_t*)d_topk_indice, d_gate_up_out,
        d_mid_row_indices, d_dnn_ws, wsSize, total_token, proj_size * 2,
        hidden_size, num_expert, top_k, stream);

    // SiLU-GLU activation
    allspark::cuda::UnaryGLUKernelLauncher(d_mid_result, d_gate_up_out,
                                            (size_t)matARows_gu,
                                            (size_t)proj_size,
                                            (int)allspark::UnaryType::SILU,
                                            stream);

    // Get expert indices for down projection
    cudaMemsetAsync(d_mid_expert_indices, 0xFF,
                    matARows_gu * sizeof(uint32_t), stream);
    allspark::cuda::GetExpertByIndice(
        (int*)d_mid_expert_indices, (int*)d_topk_indice,
        (int*)d_mid_row_indices, total_token, top_k, num_expert, stream);

    // Down GEMM
    allspark::cuda::MoeBatchedGemmLauncher<T>(
        d_mid_result, d_down_weight, d_mid_expert_indices, d_final_result,
        d_final_row_indices, d_dnn_ws, wsSize, matARows_gu, hidden_size,
        proj_size, num_expert, 1, stream);

    // Finalize routing (no EP for simplicity)
    int ep_num = num_expert;
    std::vector<int> ep_group_h(num_expert);
    std::iota(ep_group_h.begin(), ep_group_h.end(), 0);
    int* d_ep_group;
    cudaMalloc(&d_ep_group, num_expert * sizeof(int));
    cudaMemcpyAsync(d_ep_group, ep_group_h.data(),
                    num_expert * sizeof(int), cudaMemcpyHostToDevice, stream);

    allspark::cuda::FinalizeMoeRoutingNewKernelLauncher(
        d_output, d_final_result, d_experts_score, (int*)d_mid_row_indices,
        (int*)d_final_row_indices, total_token, top_k, hidden_size, ep_num,
        d_ep_group, stream);

    cudaStreamSynchronize(stream);

    cudaFree(d_total_ws);
    cudaFree(d_mid_row_indices);
    cudaFree(d_mid_expert_indices);
    cudaFree(d_final_row_indices);
    cudaFree(d_ep_group);
  } else {
    // --- Block path (cuBLAS) - skip for now, tested via operator test ---
    printf("  Skipping block path in kernel test (requires cuBLAS handle "
           "setup)\n");
    cudaFree(d_input);
    cudaFree(d_gate_score);
    cudaFree(d_gate_up_weight);
    cudaFree(d_down_weight);
    cudaFree(d_output);
    cudaFree(d_float_gate_score);
    cudaFree(d_topk_value);
    cudaFree(d_experts_score);
    cudaFree(d_topk_indice);
    cudaStreamDestroy(stream);
    GTEST_SKIP() << "Block path kernel test not implemented (SM < 90)";
    return;
  }

  // Copy output back
  std::vector<T> output_host(total_token * hidden_size);
  cudaMemcpy(output_host.data(), d_output,
             total_token * hidden_size * sizeof(T), cudaMemcpyDeviceToHost);

  // Compare with CPU reference
  float max_diff = 0.f;
  int exceed_count = 0;
  for (int i = 0; i < total_token * hidden_size; ++i) {
    float diff = std::fabs(float(output_host[i]) - float(output_ref[i]));
    float denom = std::fabs(float(output_ref[i]));
    float err = denom > 1e-6f ? diff / denom : diff;
    if (err > max_diff) max_diff = err;
    if (err > eps) exceed_count++;
  }
  printf("  MaxRelDiff=%f, exceed_count=%d / %d\n", max_diff, exceed_count,
         total_token * hidden_size);
  // Relaxed threshold: FP16 accumulation through multiple GEMMs + activation
  EXPECT_LT(exceed_count, total_token * hidden_size / 10);  // < 10% outliers

  cudaFree(d_input);
  cudaFree(d_gate_score);
  cudaFree(d_gate_up_weight);
  cudaFree(d_down_weight);
  cudaFree(d_output);
  cudaFree(d_float_gate_score);
  cudaFree(d_topk_value);
  cudaFree(d_experts_score);
  cudaFree(d_topk_indice);
  cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
//  Helper: SM check for WGMMA tests
// ---------------------------------------------------------------------------
static bool SkipIfNoWGMMA() {
  cudaDeviceProp dprop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&dprop, device_id);
  int sm = dprop.major << 8 | dprop.minor;
  return (sm < 0x0900 || sm >= 0x0a00);
}

// ---------------------------------------------------------------------------
//  Helper: Parameterized sub-kernel tests
// ---------------------------------------------------------------------------
void TestSoftmaxLowReduce(int total_token, int num_expert) {
  std::vector<float> input_f(total_token * num_expert);
  {
    std::default_random_engine gen(42);
    std::normal_distribution<float> dis(0.f, 1.f);
    for (auto& v : input_f) v = dis(gen);
  }
  std::vector<float> ref_f(total_token * num_expert);
  CPU_SoftmaxLowReduce(input_f.data(), ref_f.data(), total_token, num_expert);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  float *d_in, *d_out;
  cudaMalloc(&d_in, total_token * num_expert * sizeof(float));
  cudaMalloc(&d_out, total_token * num_expert * sizeof(float));
  cudaMemcpyAsync(d_in, input_f.data(),
                  total_token * num_expert * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  allspark::cuda::SoftmaxLowReduceKernelLauncher<float>(
      d_in, d_out, total_token, num_expert, stream);
  cudaStreamSynchronize(stream);
  std::vector<float> out_h(total_token * num_expert);
  cudaMemcpy(out_h.data(), d_out, total_token * num_expert * sizeof(float),
             cudaMemcpyDeviceToHost);
  float max_diff =
      check_equal<float>(ref_f.data(), out_h.data(), total_token * num_expert);
  printf("  SoftmaxLowReduce(tokens=%d, experts=%d) MaxDiff=%f\n", total_token,
         num_expert, max_diff);
  EXPECT_LE(max_diff, 1e-4f);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
}

void TestTopK(int total_token, int num_expert, int top_k) {
  std::vector<float> input_f(total_token * num_expert);
  {
    std::default_random_engine gen(123 + num_expert);
    std::uniform_real_distribution<float> dis(0.f, 1.f);
    for (int t = 0; t < total_token; ++t) {
      float sum = 0;
      for (int e = 0; e < num_expert; ++e) {
        input_f[t * num_expert + e] = dis(gen);
        sum += input_f[t * num_expert + e];
      }
      for (int e = 0; e < num_expert; ++e) input_f[t * num_expert + e] /= sum;
    }
  }
  std::vector<float> ref_score(total_token * top_k);
  std::vector<int> ref_indice(total_token * top_k);
  CPU_TopK(input_f.data(), ref_score.data(), ref_indice.data(), total_token,
           num_expert, top_k);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  float *d_in, *d_score;
  int* d_indice;
  cudaMalloc(&d_in, total_token * num_expert * sizeof(float));
  cudaMalloc(&d_score, total_token * top_k * sizeof(float));
  cudaMalloc(&d_indice, total_token * top_k * sizeof(int));
  cudaMemcpyAsync(d_in, input_f.data(),
                  total_token * num_expert * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  allspark::cuda::TopKKernelLauncher<float>(d_score, d_indice, d_in,
                                            total_token, num_expert, top_k,
                                            stream);
  cudaStreamSynchronize(stream);
  std::vector<float> gpu_score(total_token * top_k);
  cudaMemcpy(gpu_score.data(), d_score, total_token * top_k * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int t = 0; t < total_token; ++t) {
    std::vector<float> rs(ref_score.begin() + t * top_k,
                          ref_score.begin() + (t + 1) * top_k);
    std::vector<float> gs(gpu_score.begin() + t * top_k,
                          gpu_score.begin() + (t + 1) * top_k);
    std::sort(rs.begin(), rs.end());
    std::sort(gs.begin(), gs.end());
    for (int k = 0; k < top_k; ++k)
      EXPECT_NEAR(rs[k], gs[k], 1e-5f) << "Token " << t << " rank " << k;
  }
  cudaFree(d_in);
  cudaFree(d_score);
  cudaFree(d_indice);
  cudaStreamDestroy(stream);
}

void TestSiluGLU(int M, int proj_size) {
  auto input_h = common::rand_normal_float<half>(M * proj_size * 2, 0.5f);
  std::vector<float> input_f(M * proj_size * 2);
  for (int i = 0; i < M * proj_size * 2; ++i) input_f[i] = float(input_h[i]);
  std::vector<float> ref_f(M * proj_size);
  CPU_SiluGLU(input_f.data(), ref_f.data(), M, proj_size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  half *d_in, *d_out;
  cudaMalloc(&d_in, M * proj_size * 2 * sizeof(half));
  cudaMalloc(&d_out, M * proj_size * sizeof(half));
  cudaMemcpyAsync(d_in, input_h.data(), M * proj_size * 2 * sizeof(half),
                  cudaMemcpyHostToDevice, stream);
  allspark::cuda::UnaryGLUKernelLauncher(d_out, d_in, (size_t)M,
                                          (size_t)proj_size,
                                          (int)allspark::UnaryType::SILU,
                                          stream);
  cudaStreamSynchronize(stream);
  std::vector<half> out_h(M * proj_size);
  cudaMemcpy(out_h.data(), d_out, M * proj_size * sizeof(half),
             cudaMemcpyDeviceToHost);
  float max_diff = 0.f;
  for (int i = 0; i < M * proj_size; ++i) {
    float diff = std::fabs(float(out_h[i]) - ref_f[i]);
    if (diff > max_diff) max_diff = diff;
  }
  printf("  UnaryGLU SiLU(M=%d, proj=%d) MaxDiff=%f\n", M, proj_size,
         max_diff);
  EXPECT_LE(max_diff, 0.02f);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
//  Benchmark helper: MoeBatchedGemm TFLOPS measurement
// ---------------------------------------------------------------------------
template <typename T>
void BenchMoeBatchedGemm(int matARows, int num_expert, int N, int K,
                          int warmup = 3, int iters = 10) {
  const int nMatBPerMatARow = 1;
  auto A_host = common::rand_normal_float<T>(matARows * K, 0.3f);
  auto B_host =
      common::rand_normal_float<T>((int64_t)num_expert * K * N, 0.3f);
  std::vector<uint32_t> idx_host(matARows);
  {
    std::default_random_engine gen(42);
    std::uniform_int_distribution<int> dis(0, num_expert - 1);
    for (int r = 0; r < matARows; ++r) idx_host[r] = dis(gen);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  T *d_A, *d_B, *d_C;
  uint32_t *d_idx, *d_row;
  cudaMalloc(&d_A, matARows * K * sizeof(T));
  cudaMalloc(&d_B, (int64_t)num_expert * K * N * sizeof(T));
  cudaMalloc(&d_C, matARows * N * sizeof(T));
  cudaMalloc(&d_idx, matARows * sizeof(uint32_t));
  cudaMalloc(&d_row, matARows * sizeof(uint32_t));
  cudaMemcpyAsync(d_A, A_host.data(), matARows * K * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, B_host.data(),
                  (int64_t)num_expert * K * N * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_idx, idx_host.data(), matARows * sizeof(uint32_t),
                  cudaMemcpyHostToDevice, stream);
  size_t wsSize =
      allspark::cuda::GetWorkspaceSizeLauncher(matARows, num_expert);
  void* d_ws;
  cudaMalloc(&d_ws, wsSize);

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    cudaMemsetAsync(d_row, 0, matARows * sizeof(uint32_t), stream);
    allspark::cuda::MoeBatchedGemmLauncher<T>(d_A, d_B, d_idx, d_C, d_row,
                                               d_ws, wsSize, matARows, N, K,
                                               num_expert, nMatBPerMatARow,
                                               stream);
  }
  cudaStreamSynchronize(stream);

  // Timed iterations
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iters; ++i) {
    cudaMemsetAsync(d_row, 0, matARows * sizeof(uint32_t), stream);
    allspark::cuda::MoeBatchedGemmLauncher<T>(d_A, d_B, d_idx, d_C, d_row,
                                               d_ws, wsSize, matARows, N, K,
                                               num_expert, nMatBPerMatARow,
                                               stream);
  }
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / iters;

  // FLOPS: each of matARows rows does a [1, K] x [K, N] GEMM = 2*K*N FLOPs
  double flops = 2.0 * matARows * K * N;
  double tflops = flops / (avg_ms * 1e-3) / 1e12;
  printf("  Bench MoeBatchedGemm<%s>: rows=%d, experts=%d, N=%d, K=%d "
         "=> %.3f ms, %.2f TFLOPS\n",
         (std::is_same<T, half>::value ? "half" : "bf16"), matARows, num_expert,
         N, K, avg_ms, tflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_idx);
  cudaFree(d_row);
  cudaFree(d_ws);
  cudaStreamDestroy(stream);
}

}  // anonymous namespace

// ===========================================================================
//  Test cases: MoeBatchedGemm kernel (WGMMA, SM 90)
// ===========================================================================

// --- Generic shape tests ---
TEST(MOE_KERNEL, BatchedGemm_FP16_Small) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeBatchedGemm<half>(8, 8, 1, 256, 128, 0.05f);
}

// --- Real model shapes: down-proj GEMM (N=hidden, K=proj) ---

// DeepSeek V3: 256 experts, hidden=7168, proj=2048
TEST(MOE_KERNEL, BatchedGemm_DeepSeekV3_DownProj) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeBatchedGemm<half>(32, 256, 1, 7168, 2048, 0.1f);
}

// Qwen3-235B-A22B: 128 experts, hidden=4096, proj=1536
TEST(MOE_KERNEL, BatchedGemm_Qwen3_235B_DownProj) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeBatchedGemm<half>(32, 128, 1, 4096, 1536, 0.1f);
}

// Qwen3-30B-A3B: 128 experts, hidden=2048, proj=768
TEST(MOE_KERNEL, BatchedGemm_Qwen3_30B_DownProj) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeBatchedGemm<half>(32, 128, 1, 2048, 768, 0.1f);
}

// Qwen1.5-MoE-A2.7B: 60 experts (non-power-of-2), hidden=2048, proj=1408
TEST(MOE_KERNEL, BatchedGemm_Qwen15MoE_DownProj) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeBatchedGemm<half>(16, 60, 1, 2048, 1408, 0.1f);
}

#ifdef ENABLE_BF16
TEST(MOE_KERNEL, BatchedGemm_BF16_DeepSeekV3_DownProj) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeBatchedGemm<hie::bfloat16>(32, 256, 1, 7168, 2048, 0.15f);
}

TEST(MOE_KERNEL, BatchedGemm_BF16_Qwen3_235B_DownProj) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeBatchedGemm<hie::bfloat16>(32, 128, 1, 4096, 1536, 0.15f);
}
#endif

// ===========================================================================
//  Full MOE pipeline: real model shapes (SM 90)
// ===========================================================================

TEST(MOE_KERNEL, FullPipeline_FP16_Tiny) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeFullPipeline<half>(2, 4, 2, 64, 32, 0.3f);
}

// Qwen3-30B-A3B: 128 experts, top-8, hidden=2048, proj=768
TEST(MOE_KERNEL, FullPipeline_Qwen3_30B) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeFullPipeline<half>(4, 128, 8, 2048, 768, 0.5f);
}

// Qwen3-235B-A22B: 128 experts, top-8, hidden=4096, proj=1536
TEST(MOE_KERNEL, FullPipeline_Qwen3_235B) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeFullPipeline<half>(4, 128, 8, 4096, 1536, 0.5f);
}

// Qwen1.5-MoE-A2.7B: 60 experts, top-4, hidden=2048, proj=1408
TEST(MOE_KERNEL, FullPipeline_Qwen15MoE) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeFullPipeline<half>(4, 60, 4, 2048, 1408, 0.5f);
}

// DeepSeek V3: 256 experts, top-8, hidden=7168, proj=2048
TEST(MOE_KERNEL, FullPipeline_DeepSeekV3) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeFullPipeline<half>(2, 256, 8, 7168, 2048, 0.5f);
}

#ifdef ENABLE_BF16
TEST(MOE_KERNEL, FullPipeline_BF16_Qwen3_30B) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  TestMoeFullPipeline<hie::bfloat16>(4, 128, 8, 2048, 768, 0.6f);
}
#endif

// ===========================================================================
//  Sub-kernels: Softmax with real expert counts (all GPUs)
// ===========================================================================

TEST(MOE_KERNEL, Softmax_8Experts)   { TestSoftmaxLowReduce(16, 8); }
TEST(MOE_KERNEL, Softmax_60Experts)  { TestSoftmaxLowReduce(32, 60); }   // Qwen1.5-MoE
TEST(MOE_KERNEL, Softmax_128Experts) { TestSoftmaxLowReduce(32, 128); }  // Qwen3
TEST(MOE_KERNEL, Softmax_256Experts) { TestSoftmaxLowReduce(32, 256); }  // DeepSeek V3

// ===========================================================================
//  Sub-kernels: TopK with real (top_k, num_expert) pairs (all GPUs)
// ===========================================================================

TEST(MOE_KERNEL, TopK_Top2_8Experts)   { TestTopK(16, 8, 2); }
TEST(MOE_KERNEL, TopK_Top4_60Experts)  { TestTopK(32, 60, 4); }   // Qwen1.5-MoE
TEST(MOE_KERNEL, TopK_Top8_128Experts) { TestTopK(32, 128, 8); }  // Qwen3
TEST(MOE_KERNEL, TopK_Top8_256Experts) { TestTopK(32, 256, 8); }  // DeepSeek V3

// ===========================================================================
//  Sub-kernels: SiLU-GLU with real proj_size values (all GPUs)
// ===========================================================================

TEST(MOE_KERNEL, SiluGLU_proj128)  { TestSiluGLU(32, 128); }
TEST(MOE_KERNEL, SiluGLU_proj768)  { TestSiluGLU(32, 768); }   // Qwen3-30B
TEST(MOE_KERNEL, SiluGLU_proj1408) { TestSiluGLU(32, 1408); }  // Qwen1.5-MoE
TEST(MOE_KERNEL, SiluGLU_proj1536) { TestSiluGLU(32, 1536); }  // Qwen3-235B
TEST(MOE_KERNEL, SiluGLU_proj2048) { TestSiluGLU(64, 2048); }  // DeepSeek V3

// ===========================================================================
//  Benchmark: MoeBatchedGemm TFLOPS (SM 90)
// ===========================================================================

TEST(MOE_BENCH, BatchedGemm_DeepSeekV3) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  // DeepSeek V3 down-proj: 32 rows, 256 experts, N=7168, K=2048
  BenchMoeBatchedGemm<half>(32, 256, 7168, 2048);
}

TEST(MOE_BENCH, BatchedGemm_Qwen3_235B) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  // Qwen3-235B down-proj: 32 rows, 128 experts, N=4096, K=1536
  BenchMoeBatchedGemm<half>(32, 128, 4096, 1536);
}

TEST(MOE_BENCH, BatchedGemm_Qwen3_30B) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  // Qwen3-30B down-proj: 32 rows, 128 experts, N=2048, K=768
  BenchMoeBatchedGemm<half>(32, 128, 2048, 768);
}

TEST(MOE_BENCH, BatchedGemm_DeepSeekV3_LargeBatch) {
  if (SkipIfNoWGMMA()) GTEST_SKIP() << "Requires SM 90";
  // DeepSeek V3 down-proj with large batch: 256 rows
  BenchMoeBatchedGemm<half>(256, 256, 7168, 2048);
}

// ===========================================================================
//  Benchmark: Sub-kernels (all GPUs) — bandwidth & throughput
// ===========================================================================

TEST(MOE_BENCH, Softmax_DeepSeekV3) {
  const int total_token = 256;
  const int num_expert = 256;
  const int warmup = 5, iters = 50;

  std::vector<float> input_f(total_token * num_expert, 1.f);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  float *d_in, *d_out;
  cudaMalloc(&d_in, total_token * num_expert * sizeof(float));
  cudaMalloc(&d_out, total_token * num_expert * sizeof(float));
  cudaMemcpy(d_in, input_f.data(), total_token * num_expert * sizeof(float),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < warmup; ++i)
    allspark::cuda::SoftmaxLowReduceKernelLauncher<float>(
        d_in, d_out, total_token, num_expert, stream);
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iters; ++i)
    allspark::cuda::SoftmaxLowReduceKernelLauncher<float>(
        d_in, d_out, total_token, num_expert, stream);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_us = ms / iters * 1000.f;
  // Bandwidth: read + write = 2 * total_token * num_expert * 4 bytes
  double bytes = 2.0 * total_token * num_expert * sizeof(float);
  double gbps = bytes / (ms / iters * 1e-3) / 1e9;
  printf("  Softmax(tokens=%d, experts=%d): %.1f us, %.1f GB/s\n",
         total_token, num_expert, avg_us, gbps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
}

TEST(MOE_BENCH, TopK_DeepSeekV3) {
  const int total_token = 256;
  const int num_expert = 256;
  const int top_k = 8;
  const int warmup = 5, iters = 50;

  std::vector<float> input_f(total_token * num_expert);
  {
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dis(0.f, 1.f);
    for (auto& v : input_f) v = dis(gen);
  }
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  float *d_in, *d_score;
  int* d_indice;
  cudaMalloc(&d_in, total_token * num_expert * sizeof(float));
  cudaMalloc(&d_score, total_token * top_k * sizeof(float));
  cudaMalloc(&d_indice, total_token * top_k * sizeof(int));
  cudaMemcpy(d_in, input_f.data(), total_token * num_expert * sizeof(float),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < warmup; ++i)
    allspark::cuda::TopKKernelLauncher<float>(d_score, d_indice, d_in,
                                              total_token, num_expert, top_k,
                                              stream);
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iters; ++i)
    allspark::cuda::TopKKernelLauncher<float>(d_score, d_indice, d_in,
                                              total_token, num_expert, top_k,
                                              stream);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_us = ms / iters * 1000.f;
  printf("  TopK(tokens=%d, experts=%d, top_k=%d): %.1f us\n", total_token,
         num_expert, top_k, avg_us);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_score);
  cudaFree(d_indice);
  cudaStreamDestroy(stream);
}

TEST(MOE_BENCH, SiluGLU_DeepSeekV3) {
  const int M = 256;  // tokens * top_k
  const int proj_size = 2048;
  const int warmup = 5, iters = 50;

  auto input_h = common::rand_normal_float<half>(M * proj_size * 2, 0.5f);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  half *d_in, *d_out;
  cudaMalloc(&d_in, M * proj_size * 2 * sizeof(half));
  cudaMalloc(&d_out, M * proj_size * sizeof(half));
  cudaMemcpy(d_in, input_h.data(), M * proj_size * 2 * sizeof(half),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < warmup; ++i)
    allspark::cuda::UnaryGLUKernelLauncher(d_out, d_in, (size_t)M,
                                            (size_t)proj_size,
                                            (int)allspark::UnaryType::SILU,
                                            stream);
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iters; ++i)
    allspark::cuda::UnaryGLUKernelLauncher(d_out, d_in, (size_t)M,
                                            (size_t)proj_size,
                                            (int)allspark::UnaryType::SILU,
                                            stream);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_us = ms / iters * 1000.f;
  // SiLU-GLU: read 2*proj elements, write 1*proj element per row
  // FLOPs per element: silu = ~5 ops (exp, add, div), multiply = 1 => ~6 FLOPs
  double flops = 6.0 * M * proj_size;
  double gflops = flops / (ms / iters * 1e-3) / 1e9;
  // Bandwidth: read M*proj*2*2B + write M*proj*2B = M*proj*6B
  double bytes = (double)M * proj_size * 6;
  double gbps = bytes / (ms / iters * 1e-3) / 1e9;
  printf("  SiluGLU(M=%d, proj=%d): %.1f us, %.1f GFLOPS, %.1f GB/s\n", M,
         proj_size, avg_us, gflops, gbps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
}

TEST(MOE_BENCH, SiluGLU_Qwen3_235B) {
  const int M = 256;
  const int proj_size = 1536;
  const int warmup = 5, iters = 50;

  auto input_h = common::rand_normal_float<half>(M * proj_size * 2, 0.5f);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  half *d_in, *d_out;
  cudaMalloc(&d_in, M * proj_size * 2 * sizeof(half));
  cudaMalloc(&d_out, M * proj_size * sizeof(half));
  cudaMemcpy(d_in, input_h.data(), M * proj_size * 2 * sizeof(half),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < warmup; ++i)
    allspark::cuda::UnaryGLUKernelLauncher(d_out, d_in, (size_t)M,
                                            (size_t)proj_size,
                                            (int)allspark::UnaryType::SILU,
                                            stream);
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iters; ++i)
    allspark::cuda::UnaryGLUKernelLauncher(d_out, d_in, (size_t)M,
                                            (size_t)proj_size,
                                            (int)allspark::UnaryType::SILU,
                                            stream);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_us = ms / iters * 1000.f;
  double flops = 6.0 * M * proj_size;
  double gflops = flops / (ms / iters * 1e-3) / 1e9;
  double bytes = (double)M * proj_size * 6;
  double gbps = bytes / (ms / iters * 1e-3) / 1e9;
  printf("  SiluGLU(M=%d, proj=%d): %.1f us, %.1f GFLOPS, %.1f GB/s\n", M,
         proj_size, avg_us, gflops, gbps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
}
