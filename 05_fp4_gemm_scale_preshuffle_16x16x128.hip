#include <hip/hip_runtime.h>

using fp4x2_t = unsigned char;
using fp4x32_t = fp4x2_t __attribute__((ext_vector_type(16)));
using fp32x4_t = float __attribute__((ext_vector_type(4)));
using fp32x8_t = float __attribute__((ext_vector_type(8)));

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
    const fp32x4_t a_words = __builtin_bit_cast(fp32x4_t, a_reg);
    const fp32x4_t b_words = __builtin_bit_cast(fp32x4_t, b_reg);
    const fp32x8_t a_arg{a_words[0], a_words[1], a_words[2], a_words[3], 0, 0, 0, 0};
    const fp32x8_t b_arg{b_words[0], b_words[1], b_words[2], b_words[3], 0, 0, 0, 0};

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

    const int a_row = M_HALF * 16 + lane_mn; // 找到row
    const int k_base = K_HALF * 128 + lane_k * 32; // 起始的k. 想一下单个是128 fp4. 每个lane用32个fp4.
    const fp4x2_t* ldg_a = A + a_row * 128 + k_base / 2; // 然后我们会用char, 所以要/2.
    a_reg = __builtin_bit_cast(fp4x32_t, *reinterpret_cast<const fp32x4_t*>(ldg_a)); // 16B vector read.

    const int b_col = N_HALF * 16 + lane_mn;
    const fp4x2_t* ldg_b = B + b_col * 128 + k_base / 2; // B is packed col-major [N, K/2].
    b_reg = __builtin_bit_cast(fp4x32_t, *reinterpret_cast<const fp32x4_t*>(ldg_b)); // 16B vector read.

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
__device__ __forceinline__ void store_16x16x128_subtile(
    float* C,
    const fp32x4_t c_reg,
    const int lane_mn,
    const int lane_k
) {
    // C是[16,16]的. lanex是[lane_k*4, lane_mn] 开始的 [1,4]的矩阵.
    // 这个store就很自然了.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c_row = M_HALF * 16 + lane_k * 4 + i;
        const int c_col = N_HALF * 16 + lane_mn;
        C[c_row * 32 + c_col] = c_reg[i];
    }
}

template <int M_HALF, int N_HALF>
__device__ __forceinline__ void compute_16x16x256_output_tile_preshuffled_scale(
    const fp4x2_t* A,
    const fp4x2_t* B,
    const unsigned int a_scale_word,
    const unsigned int b_scale_word,
    float* C,
    const int lane_mn,
    const int lane_k
) {
    fp32x4_t c_reg{};
    // [16,128]的A. C是[16,16] / 64 = 4 float4.
    c_reg = mfma_16x16x128_subatom_preshuffled_scale<M_HALF, N_HALF, 0>(
        A, B, a_scale_word, b_scale_word, lane_mn, lane_k, c_reg
    );
    c_reg = mfma_16x16x128_subatom_preshuffled_scale<M_HALF, N_HALF, 1>(
        A, B, a_scale_word, b_scale_word, lane_mn, lane_k, c_reg
    );
    store_16x16x128_subtile<M_HALF, N_HALF>(C, c_reg, lane_mn, lane_k);
}

__global__ void mfma_fp32_32x32x256_fp4_fp4_scale_preshuffle_16x16x128(
    const fp4x2_t* A, // 原始的 col layout, 没有进行preshuffle还
    const fp4x2_t* B,
    const unsigned int* AScale,
    const unsigned int* BScale,
    float* C
) {
    const int tid = threadIdx.x;
    const int lane_mn = tid & 15; // 我们的AB layout 是[16,4] 这样来切分thread的. [lane_mn, lane_k]
    const int lane_k = tid >> 4;
    const int word_idx = lane_k * 16 + lane_mn; // 我们preshuffle后scale是[4,16]的，正好反过来, [lane_k, lanemn]
    const unsigned int a_scale_word = AScale[word_idx];
    const unsigned int b_scale_word = BScale[word_idx];
    
    // 整体的 A是 [32,256]的. C是[32,32]的
    // 这个里面算 [16*M_HALF, 16*N_HALF]开始的[16,16] 的C
    // 那么会用A的 [16,256] 和 B的[16,256]. 其实会有一些重复的load了.
    compute_16x16x256_output_tile_preshuffled_scale<0, 0>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
    compute_16x16x256_output_tile_preshuffled_scale<0, 1>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
    compute_16x16x256_output_tile_preshuffled_scale<1, 0>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
    compute_16x16x256_output_tile_preshuffled_scale<1, 1>(A, B, a_scale_word, b_scale_word, C, lane_mn, lane_k);
}
