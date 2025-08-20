// #include "sgemm.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>    // For timing
#include <omp.h>     // For OpenMP
#include <cstdlib>   // For aligned_alloc
#include <cmath>     // For std::abs
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::all_of


int main(int argc, char* argv[]) {
    long m = (argc > 1) ? std::stol(argv[1]) : 32;
    long n = (argc > 2) ? std::stol(argv[2]) : 4096;
    long k = (argc > 3) ? std::stol(argv[3]) : 1024;
    int num_threads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads();
    int num_runs = (argc > 5) ? std::stoi(argv[5]) : 10;

    printf("Benchmarking 'llamafile_sgemm' (Dense GEMM: C=A^T*B)\n");
    printf("Matrix Dims: m=%ld, n=%ld, k=%ld\n", m, n, k);
    printf("Threads: %d\n", num_threads);
    printf("Runs for Average: %d\n\n", num_runs);

    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 1024,  // 1 GB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);

    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(num_threads);
    struct ggml_threadpool * tp = ggml_threadpool_new(
        &tpp
    );

    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor *out = ggml_mul_mat(ctx, b, a);
    ggml_build_forward_expand(gf, out);

    struct ggml_cplan cplan = ggml_graph_plan(gf, num_threads, tp);

    std::vector<uint8_t> work_data(cplan.work_size);
    cplan.work_data = work_data.data();

    ggml_graph_compute(gf, &cplan);

    // --- Timed Benchmark ---
    printf("Starting timed benchmark...\n");
    std::vector<double> durations_ms;
    for (int i = 0; i < num_runs; ++i) {
        // Reset C matrix before each run

        auto start_time = std::chrono::high_resolution_clock::now();

        ggml_graph_compute(gf, &cplan);

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

    return 0;
}
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
// #include <vector>
// #include <chrono>
// #include <iostream>


// int main(int argc, char *argv[]) {

//     struct ggml_init_params params = {
//         .mem_size = 1024 * 1024 * 1024,  // 1 GB
//         .mem_buffer = NULL,
//         .no_alloc = false,
//     };
//     struct ggml_context *ctx = ggml_init(params);
//     struct ggml_cgraph *gf = ggml_new_graph(ctx);

//     struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 32);
//     struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
//     x = ggml_mul_mat(ctx, w, x);

//     ggml_build_forward_expand(gf, x);

//     struct ggml_threadpool_params tpp = ggml_threadpool_params_default(4);
//     struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);

//     struct ggml_cplan cplan = ggml_graph_plan(gf, 4, threadpool);

//     std::vector<uint8_t> work_data(cplan.work_size);
//     std::cout << "work_size: " << cplan.work_size << std::endl;
//     cplan.work_data = work_data.data();

//     ggml_graph_compute(gf, &cplan);

//     auto t0 = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < 10; i++) {
//         ggml_graph_compute(gf, &cplan);
//     }
//     auto t1 = std::chrono::high_resolution_clock::now();

//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
//     std::cout << "Average time per run: " << duration / 10.0 << " us" << std::endl;

//     return 0;
// }