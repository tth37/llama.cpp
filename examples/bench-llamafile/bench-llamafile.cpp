// #include "sgemm.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>    // For timing
#include <omp.h>     // For OpenMP
#include <cstdlib>   // For aligned_alloc
#include <cmath>     // For std::abs
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::all_of

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};

extern "C" {
bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);
}

/**
 * @brief A simple, unoptimized reference implementation for C = A^T * B.
 *
 * This function is for verification purposes. It assumes specific memory layouts
 * based on the leading dimension (ld) parameters provided in the original code,
 * which are common in libraries like GGML/llama.cpp.
 *
 * @param A (k x m), row-major.
 * @param B (k x n), column-major.
 * @param C (m x n), column-major.
 */
void reference_sgemm(long m, long n, long k,
                     const float* A, long lda, // Assumed k x m, row-major (lda=m)
                     const float* B, long ldb, // Assumed k x n, col-major (ldb=k)
                     float* C,       long ldc)      // Assumed m x n, col-major (ldc=m)
{
    // C(i, j) = sum_{l=0}^{k-1} A_T(i, l) * B(l, j)
    for (long i = 0; i < m; ++i) {      // Iterate over rows of C
        for (long j = 0; j < n; ++j) {  // Iterate over columns of C
            float sum = 0.0f;
            for (long l = 0; l < k; ++l) { // Dot product dimension
                // A_T(i, l) is A(l, i). In row-major A, this is A[l*lda + i]
                // B(l, j). In column-major B, this is B[j*ldb + l]
                sum += A[l * lda + i] * B[j * ldb + l];
            }
            // C(i, j). In column-major C, this is C[j*ldc + i]
            C[j * ldc + i] = sum;
        }
    }
}

/**
 * @brief Verifies if two matrices are element-wise equal within a tolerance.
 */
bool verify_results(long m, long n, const float* C_actual, const float* C_ref, long ldc) {
    const float epsilon = 1e-4f * n; // Tolerance proportional to accumulation size
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            long index = j * ldc + i; // Column-major access
            if (std::abs(C_actual[index] - C_ref[index]) > epsilon) {
                fprintf(stderr, "Verification FAILED at C[%ld, %ld]: actual=%.6f vs ref=%.6f\n",
                        i, j, C_actual[index], C_ref[index]);
                return false;
            }
        }
    }
    printf("Verification PASSED!\n");
    return true;
}


int main(int argc, char* argv[]) {
    // --- Parse CLI Arguments with Defaults ---
    long m = (argc > 1) ? std::stol(argv[1]) : 32;
    long n = (argc > 2) ? std::stol(argv[2]) : 4096;
    long k = (argc > 3) ? std::stol(argv[3]) : 1024;
    int num_threads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads();
    int num_runs = (argc > 5) ? std::stoi(argv[5]) : 10;

    printf("Benchmarking 'llamafile_sgemm' (Dense GEMM: C=A^T*B)\n");
    printf("Matrix Dims: m=%ld, n=%ld, k=%ld\n", m, n, k);
    printf("Threads: %d\n", num_threads);
    printf("Runs for Average: %d\n\n", num_runs);

    struct ggml_threadpool_params params = ggml_threadpool_params_default(num_threads);
    struct ggml_threadpool * tp = ggml_threadpool_new(
        &params
    );

    // Leading dimensions (strides). These imply specific memory layouts.
    // lda=m for A(k,m) suggests row-major.
    // ldb=k for B(k,n) suggests column-major.
    // ldc=m for C(m,n) suggests column-major.
    long lda = m;
    long ldb = k;
    long ldc = m;

    size_t sizeA = (size_t)k * m;
    size_t sizeB = (size_t)k * n;
    size_t sizeC = (size_t)m * n;

    float *A = (float*)aligned_alloc(64, sizeof(float) * sizeA);
    float *B = (float*)aligned_alloc(64, sizeof(float) * sizeB);
    float *C = (float*)aligned_alloc(64, sizeof(float) * sizeC);
    float *C_ref = (float*)aligned_alloc(64, sizeof(float) * sizeC);

    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices with simple values for easy verification
    for (size_t i = 0; i < sizeA; ++i) A[i] = 1.0f;
    for (size_t i = 0; i < sizeB; ++i) B[i] = 2.0f;
    memset(C, 0, sizeof(float) * sizeC);

    // --- Correctness Check ---
    printf("Computing reference result...\n");
    reference_sgemm(m, n, k, A, lda, B, ldb, C_ref, ldc);


    // --- Warm-up Run ---
    printf("Performing warm-up run...\n");
    #pragma omp parallel num_threads(num_threads)
    {
        int ith = omp_get_thread_num();
        int nth = omp_get_num_threads();
        struct ggml_compute_params params = { .ith = ith, .nth = nth, .threadpool = tp };
        llamafile_sgemm(&params, m, n, k, A, lda, B, ldb, C, ldc,
                        GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
    }
    
    // --- Timed Benchmark ---
    printf("Starting timed benchmark...\n");
    std::vector<double> durations_ms;
    for (int i = 0; i < num_runs; ++i) {
        // Reset C matrix before each run
        memset(C, 0, sizeof(float) * sizeC);

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel num_threads(num_threads)
        {
            int ith = omp_get_thread_num();
            int nth = omp_get_num_threads();
            struct ggml_compute_params params = { .ith = ith, .nth = nth, .threadpool = tp };
            llamafile_sgemm(&params, m, n, k, A, lda, B, ldb, C, ldc,
                            GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        durations_ms.push_back(std::chrono::duration<double, std::milli>(end_time - start_time).count());
    }

    double total_duration = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);
    double average_latency = total_duration / num_runs;
    double gflops = (2.0 * m * n * k) / (average_latency / 1000.0) / 1e9;

    printf("\n--- Benchmark Results ---\n");
    printf("Total time for %d runs: %.2f ms\n", num_runs, total_duration);
    printf("Average latency per run: %.3f ms\n", average_latency);
    printf("Performance: %.2f GFLOPS\n", gflops);

    printf("\n--- Correctness Verification ---\n");
    verify_results(m, n, C, C_ref, ldc);

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}
