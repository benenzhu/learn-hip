#include <hip/hip_runtime.h>

using fp4x2_t = unsigned char;
using fp4x32_t = fp4x2_t __attribute__((ext_vector_type(16)));
using fp32x4_t = float __attribute__((ext_vector_type(4)));
using fp32x16_t = float __attribute__((ext_vector_type(16)));
using int32x4_t = int __attribute__((ext_vector_type(4)));
using int32x8_t = int __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned char extract_fp4(
    const fp4x2_t x,
    const int index
) {
    return index == 0 ? (x & 0xFu) : (x >> 4);
}

__device__ __forceinline__ fp4x2_t create_fp4x2(
    const unsigned char x,
    const unsigned char y
) {
    return x | (y << 4);
}

template <int OPSEL_A, int OPSEL_B>
__device__ __forceinline__ fp32x4_t mfma_16x16x128_fp4_fp4_opsel(
    const fp4x32_t a_reg,
    const fp4x32_t b_reg,
    const fp32x4_t c_reg,
    const unsigned int scale_a,
    const unsigned int scale_b
) {
    const int32x4_t a_words = __builtin_bit_cast(int32x4_t, a_reg);
    const int32x4_t b_words = __builtin_bit_cast(int32x4_t, b_reg);
    const int32x8_t a_arg{a_words[0], a_words[1], a_words[2], a_words[3], 0, 0, 0, 0};
    const int32x8_t b_arg{b_words[0], b_words[1], b_words[2], b_words[3], 0, 0, 0, 0};

    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a_arg,
        b_arg,
        c_reg,
        4, // A format: fp4 e2m1
        4, // B format: fp4 e2m1
        OPSEL_A,
        scale_a,
        OPSEL_B,
        scale_b
    );
}

__device__ __forceinline__ fp32x4_t mfma_16x16x128_fp4_fp4(
    const fp4x32_t a_reg,
    const fp4x32_t b_reg,
    const fp32x4_t c_reg,
    const unsigned int scale_a,
    const unsigned int scale_b
) {
    return mfma_16x16x128_fp4_fp4_opsel<0, 0>(
        a_reg,
        b_reg,
        c_reg,
        scale_a,
        scale_b
    );
}

template <int M_HALF, int N_HALF, int K_HALF>
__device__ __forceinline__ fp32x4_t mfma_16x16x128_subatom_preshuffled_scale(
    const fp4x2_t* A,
    const fp4x2_t* B,
    const unsigned int a_scale_word,
    const unsigned int b_scale_word,
    const int lane_mn,
    const int lane_k,
    const fp32x4_t c_reg
) {
    fp4x32_t a_reg{};
    fp4x32_t b_reg{};

    const int a_row = M_HALF * 16 + lane_mn;
    const int k_base = K_HALF * 128 + lane_k * 32;
    const fp4x2_t* ldg_a = A + a_row * 128 + k_base / 2;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        a_reg[i] = ldg_a[i];
    }

    const int b_col = N_HALF * 16 + lane_mn;
    const fp4x2_t* ldg_b = B + k_base * 16 + b_col / 2;
    const int b_extract_idx = b_col & 1;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const unsigned char lo = extract_fp4(ldg_b[16 * (2 * i)], b_extract_idx);
        const unsigned char hi = extract_fp4(ldg_b[16 * (2 * i + 1)], b_extract_idx);
        b_reg[i] = create_fp4x2(lo, hi);
    }

    constexpr int A_SCALE_BYTE = M_HALF + 2 * K_HALF;
    constexpr int B_SCALE_BYTE = N_HALF + 2 * K_HALF;

    return mfma_16x16x128_fp4_fp4_opsel<A_SCALE_BYTE, B_SCALE_BYTE>(
        a_reg,
        b_reg,
        c_reg,
        a_scale_word,
        b_scale_word
    );
}

template <int M_HALF, int N_HALF>
__device__ __forceinline__ void store_16x16_subtile(
    float* C,
    const fp32x4_t c_reg,
    const int lane_mn,
    const int lane_k
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c_row = M_HALF * 16 + lane_k * 4 + i;
        const int c_col = N_HALF * 16 + lane_mn;
        C[c_row * 32 + c_col] = c_reg[i];
    }
}

template <int M_HALF, int N_HALF>
__device__ __forceinline__ void compute_16x16_output_tile_preshuffled_scale(
    const fp4x2_t* A,
    const fp4x2_t* B,
    const unsigned int a_scale_word,
    const unsigned int b_scale_word,
    float* C,
    const int lane_mn,
    const int lane_k
) {
    fp32x4_t c_reg{};
    c_reg = mfma_16x16x128_subatom_preshuffled_scale<M_HALF, N_HALF, 0>(
        A, B, a_scale_word, b_scale_word, lane_mn, lane_k, c_reg
    );
    c_reg = mfma_16x16x128_subatom_preshuffled_scale<M_HALF, N_HALF, 1>(
        A, B, a_scale_word, b_scale_word, lane_mn, lane_k, c_reg
    );
    store_16x16_subtile<M_HALF, N_HALF>(C, c_reg, lane_mn, lane_k);
}

__global__ void mfma_fp32_32x32x64_fp4_fp4(
    const fp4x2_t* A,
    const fp4x2_t* B,
    float* C
) {
    const int tid = threadIdx.x;
    const int lane_n = tid % 32;
    const int lane_k = tid / 32;

    fp4x32_t a_reg{};
    fp4x32_t b_reg{};
    fp32x16_t c_reg{};

    // A is logical [32, 64] row-major fp4, packed as [32, 32] fp4x2.
    // lane 0 loads A[0, 0:32], lane 32 loads A[0, 32:64].
    const fp4x2_t* ldg_a = A + lane_n * 32 + lane_k * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        a_reg[i] = ldg_a[i];
    }

    // B is laid out for the MFMA B fragment. Adjacent lanes select low/high
    // nibbles from the same byte, then repack two K positions into fp4x2.
    const fp4x2_t* ldg_b = B + lane_n / 2 + 16 * 32 * lane_k;
    const int b_extract_idx = lane_n % 2;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const unsigned char lo = extract_fp4(ldg_b[16 * (2 * i)], b_extract_idx);
        const unsigned char hi = extract_fp4(ldg_b[16 * (2 * i + 1)], b_extract_idx);
        b_reg[i] = create_fp4x2(lo, hi);
    }

    const int32x4_t a_words = __builtin_bit_cast(int32x4_t, a_reg);
    const int32x4_t b_words = __builtin_bit_cast(int32x4_t, b_reg);
    const int32x8_t a_arg{a_words[0], a_words[1], a_words[2], a_words[3], 0, 0, 0, 0};
    const int32x8_t b_arg{b_words[0], b_words[1], b_words[2], b_words[3], 0, 0, 0, 0};

    const unsigned int scale_a = 127u; // e8m0 scale 2^(127 - 127) = 1.0
    const unsigned int scale_b = 127u;
    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_arg,
        b_arg,
        c_reg,
        4, // A format: fp4 e2m1
        4, // B format: fp4 e2m1
        0,
        scale_a,
        0,
        scale_b
    );

    // The 16 C registers per lane cover a 32x32 output tile.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        C[lane_n + (lane_k * 4 + 0) * 32 + i * 32 * 8] = c_reg[i * 4 + 0];
        C[lane_n + (lane_k * 4 + 1) * 32 + i * 32 * 8] = c_reg[i * 4 + 1];
        C[lane_n + (lane_k * 4 + 2) * 32 + i * 32 * 8] = c_reg[i * 4 + 2];
        C[lane_n + (lane_k * 4 + 3) * 32 + i * 32 * 8] = c_reg[i * 4 + 3];
    }
}

__global__ void mfma_fp32_32x32x256_fp4_fp4_scale_preshuffle(
    const fp4x2_t* A,
    const fp4x2_t* B,
    const unsigned int* AScale,
    const unsigned int* BScale,
    float* C
) {
    const int tid = threadIdx.x;
    const int lane_mn = tid & 15;
    const int lane_k = tid >> 4;
    const int word_idx = lane_k * 16 + lane_mn;
    const unsigned int a_scale_word = AScale[word_idx];
    const unsigned int b_scale_word = BScale[word_idx];

    compute_16x16_output_tile_preshuffled_scale<0, 0>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
    compute_16x16_output_tile_preshuffled_scale<0, 1>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
    compute_16x16_output_tile_preshuffled_scale<1, 0>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
    compute_16x16_output_tile_preshuffled_scale<1, 1>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
}

__global__ void mfma_fp32_16x16x128_fp4_fp4(
    const fp4x2_t* A,
    const fp4x2_t* B,
    const fp4x2_t* AScale,
    const fp4x2_t* BScale,
    float* C
) {
    const int tid = threadIdx.x;
    const int lane_mn = tid & 15;
    const int lane_k = tid >> 4;

    fp4x32_t a_reg{};
    fp4x32_t b_reg{};
    fp32x4_t c_reg{};

    // A is logical [16, 128] row-major fp4, packed as [16, 64] fp4x2.
    // kABKLane=4, so each lane owns 32 fp4 values from one K quarter.
    const fp4x2_t* ldg_a = A + lane_mn * 64 + lane_k * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        a_reg[i] = ldg_a[i];
    }

    // B is logical [128, 16]. For each K row, adjacent N columns are packed
    // into low/high nibbles. Each lane extracts one N column across 32 K values.
    const fp4x2_t* ldg_b = B + lane_k * 32 * 8 + lane_mn / 2;
    const int b_extract_idx = lane_mn & 1;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const unsigned char lo = extract_fp4(ldg_b[8 * (2 * i)], b_extract_idx);
        const unsigned char hi = extract_fp4(ldg_b[8 * (2 * i + 1)], b_extract_idx);
        b_reg[i] = create_fp4x2(lo, hi);
    }

    const int32x4_t a_words = __builtin_bit_cast(int32x4_t, a_reg);
    const int32x4_t b_words = __builtin_bit_cast(int32x4_t, b_reg);
    const int32x8_t a_arg{a_words[0], a_words[1], a_words[2], a_words[3], 0, 0, 0, 0};
    const int32x8_t b_arg{b_words[0], b_words[1], b_words[2], b_words[3], 0, 0, 0, 0};

    // With opselA=opselB=0, the instruction consumes the low byte of each
    // scale operand. kABKLane=4, so each lane supplies the scale for its own
    // 32-wide K quarter.
    const unsigned int scale_a = AScale[lane_k * 16 + lane_mn];
    const unsigned int scale_b = BScale[lane_k * 16 + lane_mn];
    c_reg = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a_arg,
        b_arg,
        c_reg,
        4, // A format: fp4 e2m1
        4, // B format: fp4 e2m1
        0,
        scale_a,
        0,
        scale_b
    );

    // kCMLane=4, kCNLane=16: each lane writes four rows for one N column.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        C[(lane_k * 4 + i) * 16 + lane_mn] = c_reg[i];
    }
}

__global__ void mfma_fp32_32x32x256_fp4_fp4_packed_scale(
    const fp4x2_t* A,
    const fp4x2_t* B,
    const unsigned int* AScale,
    const unsigned int* BScale,
    float* C
) {
    const int tid = threadIdx.x;
    const int lane_mn = tid & 15;
    const int lane_k = tid >> 4;

    // This tile is built from 2(M halves) x 2(N halves) x 2(K halves)
    // 16x16x128 atoms. A/B scales are packed as:
    //   byte0 = half0, kg0..3
    //   byte1 = half1, kg0..3
    //   byte2 = half0, kg4..7
    //   byte3 = half1, kg4..7
    #pragma unroll
    for (int m_half = 0; m_half < 2; ++m_half) {
        #pragma unroll
        for (int n_half = 0; n_half < 2; ++n_half) {
            fp32x4_t c_reg{};

            #pragma unroll
            for (int k_half = 0; k_half < 2; ++k_half) {
                fp4x32_t a_reg{};
                fp4x32_t b_reg{};

                const int a_row = m_half * 16 + lane_mn;
                const int k_base = k_half * 128 + lane_k * 32;
                const fp4x2_t* ldg_a = A + a_row * 128 + k_base / 2;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    a_reg[i] = ldg_a[i];
                }

                const int b_col = n_half * 16 + lane_mn;
                const fp4x2_t* ldg_b = B + k_base * 16 + b_col / 2;
                const int b_extract_idx = b_col & 1;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const unsigned char lo = extract_fp4(ldg_b[16 * (2 * i)], b_extract_idx);
                    const unsigned char hi = extract_fp4(ldg_b[16 * (2 * i + 1)], b_extract_idx);
                    b_reg[i] = create_fp4x2(lo, hi);
                }

                const int word_idx = lane_k * 16 + lane_mn;
                const int a_byte_idx = m_half + 2 * k_half;
                const int b_byte_idx = n_half + 2 * k_half;
                const unsigned int scale_a = (AScale[word_idx] >> (8 * a_byte_idx)) & 0xFFu;
                const unsigned int scale_b = (BScale[word_idx] >> (8 * b_byte_idx)) & 0xFFu;
                c_reg = mfma_16x16x128_fp4_fp4(a_reg, b_reg, c_reg, scale_a, scale_b);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c_row = m_half * 16 + lane_k * 4 + i;
                const int c_col = n_half * 16 + lane_mn;
                C[c_row * 32 + c_col] = c_reg[i];
            }
        }
    }
}
