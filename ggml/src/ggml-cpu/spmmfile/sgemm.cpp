#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"

// if we are on arm
#ifdef __ARM_ARCH
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
#else
bool llamafile_sgemm_sparse_chunked_arm80(
    long m, long n, long k,
    const float *A, long lda,
    const float *B, long ldb,
    float *C, long ldc,
    float threshold,
    int sub_batches,
    int ith, int nth) {
    printf("ARM sgemm_sparse_chunked_arm80\n");
    return false;
}
#endif // __ARM_ARCH



bool spmmfile_sgemm(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     const void *A, int64_t lda, const void *B, int64_t ldb, void *C,
                     int64_t ldc, int Atype, int Btype, int Ctype) {
    
    // if (params->ith == 0) {
    //     printf("params->ith=%d, params->nth=%d\n", params->ith, params->nth);
    //     printf("m=%d, n=%d, k=%d\n\n", m, n, k);
    // }
    // return true;
    if (Atype != GGML_TYPE_F32 || Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32) {
        fprintf(stderr, "spmmfile_sgemm: only GGML_TYPE_F32 is supported\n");
        return false;
    }
    
    return llamafile_sgemm_sparse_chunked_arm80(
        m, n, k,
        (const float *)A, lda,
        (const float *)B, ldb,
        (float *)C, ldc,
        0.0f, // threshold
        1,    // sub_batches
        params->ith, params->nth
    );

}


