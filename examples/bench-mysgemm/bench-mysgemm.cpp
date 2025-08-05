#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <cstring>
#include <random>
#include <arm_neon.h>
#include <omp.h>

bool llamafile_sgemm_sparse_chunked_arm80(
    long m, long n, long k,
    const float *A, long lda,
    const float *B, long ldb,
    float *C, long ldc,
    float threshold,
    int sub_batches,
    int ith, int nth);

void reference_sgemm_sparse_chunked(long m, long n, long k,
                                  const float* A, long lda,
                                  const float* B, long ldb,
                                  float* C, long ldc,
                                  float threshold, int sub_batches)
{
    long m_per_batch = (m + sub_batches - 1) / sub_batches;

    // Phase 1: Build the activation map, same as the kernel
    std::vector<std::vector<bool>> activation_map(sub_batches, std::vector<bool>(k, false));
    for (int batch_idx = 0; batch_idx < sub_batches; ++batch_idx) {
        long m_start = batch_idx * m_per_batch;
        long m_end = std::min(m, (batch_idx + 1) * m_per_batch);
        long m_chunk_size = m_end - m_start;
        if(m_chunk_size <= 0) continue;

        for (long j = 0; j < k; ++j) {
            float sum_abs = 0.0f;
            for (long i = m_start; i < m_end; ++i) {
                sum_abs += std::abs(A[j * lda + i]);
            }
            if ((sum_abs / m_chunk_size) > threshold) {
                activation_map[batch_idx][j] = true;
            }
        }
    }

    // Phase 2: Compute C += A * B using the activation map
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            int batch_idx = i / m_per_batch;
            float dot_product = 0.0f;
            for (long l = 0; l < k; ++l) {
                if (activation_map[batch_idx][l]) {
                    // A(i, l) is A[l*lda + i]
                    // B(l, j) is B[j*ldb + l]
                    dot_product += A[l * lda + i] * B[j * ldb + l];
                }
            }
            C[j * ldc + i] += dot_product;
        }
    }
}

bool verify_results(long m, long n, const float* C_actual, const float* C_ref, long ldc) {
    const float epsilon = 1e-4f;
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            long index = j * ldc + i;
            if (std::abs(C_actual[index] - C_ref[index]) > epsilon) {
                fprintf(stderr, "Verification FAILED at C[%ld, %ld]: actual=%.6f vs ref=%.6f (diff=%.6f)\n",
                        i, j, C_actual[index], C_ref[index], std::abs(C_actual[index] - C_ref[index]));
                return false;
            }
        }
    }
    printf("Verification PASSED!\n");
    return true;
}


bool llamafile_sgemm_sparse_chunked_arm80(
    long m, long n, long k,
    const float *A, long lda,
    const float *B, long ldb,
    float *C, long ldc,
    float threshold,
    int sub_batches,
    int ith, int nth) {

    if (sub_batches <= 0) return false;

    std::vector<std::vector<long>> active_cols_per_batch(sub_batches);
    long m_per_batch = (m + sub_batches - 1) / sub_batches;

    // Phase 1: Identify active columns for each sub-batch
    for (int batch_idx = 0; batch_idx < sub_batches; ++batch_idx) {
        long m_start = batch_idx * m_per_batch;
        long m_end = std::min(m, (batch_idx + 1) * m_per_batch);
        long m_chunk_size = m_end - m_start;
        if (m_chunk_size <= 0) continue;

        active_cols_per_batch[batch_idx].reserve(k);
        for (long j = 0; j < k; ++j) {
            float sum_abs = 0.0f;
            long i = m_start;

            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (; i + 3 < m_end; i += 4) {
                 const float32x4_t a_vec = vld1q_f32(&A[j * lda + i]);
                 sum_vec = vaddq_f32(sum_vec, vabsq_f32(a_vec));
            }
            sum_abs += vaddvq_f32(sum_vec);

            for (; i < m_end; ++i) sum_abs += fabsf(A[j * lda + i]);

            if ((sum_abs / m_chunk_size) > threshold) {
                active_cols_per_batch[batch_idx].push_back(j);
            }
        }
    }

    long n_start = (n * ith) / nth;
    long n_end = (n * (ith + 1)) / nth;

    for (long j = n_start; j < n_end; ++j) {
        for (int batch_idx = 0; batch_idx < sub_batches; ++batch_idx) {
            long m_start = batch_idx * m_per_batch;
            long m_end = std::min(m, (batch_idx + 1) * m_per_batch);
            if (m_start >= m_end) continue;

            const std::vector<long>& active_indices = active_cols_per_batch[batch_idx];
            for (long k_val : active_indices) {
                const float b_scalar = B[j * ldb + k_val];
                const float32x4_t b_broadcast = vdupq_n_f32(b_scalar);
                const float* pA = &A[k_val * lda];
                float* pC = &C[j * ldc];

                long i = m_start;
                long i_vec_end = m_start + ((m_end - m_start) & ~3L);

                // Use NEON FMA (Fused Multiply-Add) for the main loop
                for (; i < i_vec_end; i += 4) {
                    const float32x4_t a_vec = vld1q_f32(pA + i);
                    float32x4_t c_vec = vld1q_f32(pC + i);
                    c_vec = vfmaq_f32(c_vec, a_vec, b_broadcast);
                    vst1q_f32(pC + i, c_vec);
                }

                for (; i < m_end; ++i) {
                    pC[i] += pA[i] * b_scalar;
                }
            }
        }
    }
    return true;
}


int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 8) {
        fprintf(stderr, "Usage: %s <m> <n> <k> <sparsity_ratio> <sub_batches> [num_threads] [num_runs]\n", argv[0]);
        fprintf(stderr, "  sparsity_ratio: a value between 0.0 and 1.0. e.g., 0.9 means 90%% sparse (10%% active chunks).\n");
        return 1;
    }
    long m = std::stol(argv[1]);
    long n = std::stol(argv[2]);
    long k = std::stol(argv[3]);
    float sparsity_ratio = std::stof(argv[4]);
    int sub_batches = std::stoi(argv[5]);
    int num_threads = (argc > 6) ? std::stoi(argv[6]) : omp_get_max_threads();
    int num_runs = (argc > 7) ? std::stoi(argv[7]) : 10;

    if (sparsity_ratio < 0.0f || sparsity_ratio > 1.0f) {
        fprintf(stderr, "Error: sparsity_ratio must be between 0.0 and 1.0\n");
        return 1;
    }

    printf("Benchmarking 'llamafile_sgemm_sparse_chunked_arm80' (C += A*B)\n");
    printf("Matrix Dims: m=%ld, n=%ld, k=%ld\n", m, n, k);
    printf("Sparsity Ratio: %.2f (%.0f%% sparse, %.0f%% active), Threshold: 1.0\n",
           sparsity_ratio, sparsity_ratio * 100.0f, (1.0f - sparsity_ratio) * 100.0f);
    printf("Sub-batches: %d, Threads: %d, Runs: %d\n\n", sub_batches, num_threads, num_runs);

    long lda = m, ldb = k, ldc = m;
    size_t sizeA = (size_t)k * m;
    size_t sizeB = (size_t)k * n;
    size_t sizeC = (size_t)m * n;

    const float INIT_C_VAL = 100.0f;
    const float INACTIVE_VAL = 0.1f;
    const float ACTIVE_VAL = 2.0f;
    const float threshold = 1.0f;

    float *A = (float*)aligned_alloc(64, sizeof(float) * sizeA);
    float *B = (float*)aligned_alloc(64, sizeof(float) * sizeB);
    float *C = (float*)aligned_alloc(64, sizeof(float) * sizeC);
    float *C_ref = (float*)aligned_alloc(64, sizeof(float) * sizeC);

    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    long m_per_batch = (m + sub_batches - 1) / sub_batches;
    long active_flops_numerator = 0;

    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    float activation_ratio = 1.0f - sparsity_ratio;

    for (long j = 0; j < k; ++j) {
        for (int batch_idx = 0; batch_idx < sub_batches; ++batch_idx) {
             long m_start = batch_idx * m_per_batch;
             long m_end = std::min(m, (batch_idx + 1) * m_per_batch);
             if (m_start >= m_end) continue;

             bool is_active = (dis(gen) < activation_ratio);

             float value_to_fill = is_active ? ACTIVE_VAL : INACTIVE_VAL;

             if (is_active) {
                 active_flops_numerator += (m_end - m_start);
             }
             for (long i = m_start; i < m_end; ++i) {
                 A[j * lda + i] = value_to_fill;
             }
        }
    }

    std::fill(B, B + sizeB, 1.0f);

    printf("--- Correctness Verification ---\n");
    printf("Computing reference result...\n");
    std::fill(C_ref, C_ref + sizeC, INIT_C_VAL);
    reference_sgemm_sparse_chunked(m, n, k, A, lda, B, ldb, C_ref, ldc, threshold, sub_batches);

    printf("Computing kernel result for verification...\n");
    std::fill(C, C + sizeC, INIT_C_VAL);
    #pragma omp parallel num_threads(num_threads)
    {
        int ith = omp_get_thread_num();
        int nth = omp_get_num_threads();
        llamafile_sgemm_sparse_chunked_arm80(m, n, k, A, lda, B, ldb, C, ldc, threshold, sub_batches, ith, nth);
    }
    verify_results(m, n, C, C_ref, ldc);
    printf("\n");


    // --- Warm-up Run ---
    printf("Performing warm-up run...\n");
    #pragma omp parallel num_threads(num_threads)
    {
        int ith = omp_get_thread_num();
        int nth = omp_get_num_threads();
        llamafile_sgemm_sparse_chunked_arm80(m, n, k, A, lda, B, ldb, C, ldc, threshold, sub_batches, ith, nth);
    }

    // --- Timed Benchmark ---
    printf("Starting timed benchmark...\n");
    std::vector<double> durations_ms;
    for (int i = 0; i < num_runs; ++i) {
        std::fill(C, C + sizeC, INIT_C_VAL);
        auto start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(num_threads)
        {
            int ith = omp_get_thread_num();
            int nth = omp_get_num_threads();
            llamafile_sgemm_sparse_chunked_arm80(m, n, k, A, lda, B, ldb, C, ldc, threshold, sub_batches, ith, nth);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        durations_ms.push_back(std::chrono::duration<double, std::milli>(end_time - start_time).count());
    }

    // --- Report Results ---
    double total_duration = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);
    double average_latency = total_duration / num_runs;
    // Calculate GFLOP/s based on the actual work done.
    // FLOPs = 2 * n * (number of active elements in a column of A) * k_active
    // The total number of active elements in A is (active_flops_numerator * k)
    // but the GFLOPS calculation here is a simplification based on active chunks.
    double gflops = (2.0 * n * active_flops_numerator) / (average_latency / 1000.0) / 1e9;

    printf("\n--- Benchmark Results ---\n");
    printf("Total time for %d runs: %.2f ms\n", num_runs, total_duration);
    printf("Average latency per run: %.3f ms\n", average_latency);
    printf("Effective Performance: %.2f GFLOPS\n", gflops);

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}