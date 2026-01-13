// empty lines for hiprtc.















/*
03_v0: 
    use 512 thread
    simple load and calculate.
    pre register allocate.
    204 VGPRs;
    Should have bank conflict problem and load problems.
- [x] use float4 load
- v0: 137 TF:
- v1: 136 TF: use float2 load for shared memory, 
    - not up. so prob mainly for the bank conflict problem?

- v2: 
-     165 TF:  remove clear shared memory.
-     162 TF: use float2 load for shared memory.
-     382 TF: add swizzle for shared memory.  (+136%)
-     396 TF: add all unroll for params.
[ ] add swizzle for shared memory.
*/







#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#ifndef PYTHON_CALL
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int BLOCK_M = 256;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int SMEM_STRIDE = 64;
constexpr int NUM_WARP_M = BLOCK_M / 64;
constexpr int NUM_WARP_N = BLOCK_N / 64;
#endif

#define cdiv(a, b) (((a) + (b) - 1) / (b))

template <int row=16, int col=16>
__device__ void print_mem(const __hip_bfloat16 *ptr){
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) { 
        for(int i = 0; i < row; i ++) {
            if(i % 8 == 0 && i != 0) {
                printf("\n");
            }
            for(int j = 0; j < col; j++) {
                if(j % 8 == 0 && j != 0) {
                    printf("  ");
                }
                
                float now = ptr[i*col+j];
                printf("%6.1lf ",  now);
            }
            printf("\n");
        }
    }
}

constexpr int THREAD_NUM = 512;

// swizzle<B, MM, S>: XOR addr 的 bit[MM+B-1:MM] 与 bit[S+B-1:S]
// Bmask = ((1 << B) - 1) << MM，选择从 bit MM 开始的 B 位
// MM = log2(load_width)，保持低 MM 位不变，确保向量加载的连续性
// 例如：float2 加载 4 个 bf16，MM=2 保持 bit[1:0] 不变
template <__hip_internal::uint32_t B, __hip_internal::uint32_t MM, __hip_internal::uint32_t S, typename T>
__device__ __forceinline__ T swizzle(T addr) {
    constexpr auto Bmask = ((1 << B) - 1) << MM;
    return ((addr >> S) & Bmask) ^ addr;
}


__launch_bounds__(THREAD_NUM)
__global__
void _3_fp16_gemm_v0(
  __hip_bfloat16 *A_gmem,
  __hip_bfloat16 *B_gmem,
  __hip_bfloat16 *C_gmem
) {
    // __hip_bfloat16 Atile[]
    
    const int tid = threadIdx.x;
    
    // g2s
    // s2r
    // doing calculate
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    
    __shared__ __hip_bfloat16 A_smem[BLOCK_M * BLOCK_K];
    __shared__ __hip_bfloat16 B_smem[BLOCK_N * BLOCK_K];
    constexpr int total_shared_size_65536 = sizeof(A_smem) + sizeof(B_smem);
    constexpr int total_iter_num = cdiv(K, BLOCK_K);
    
    constexpr int C_reg_num__128 = BLOCK_M * BLOCK_N / 512;
    constexpr int warp_reg_num__8192 = C_reg_num__128 * 64;
    float C_reg0[C_reg_num__128/2];
    float C_reg1[C_reg_num__128/2];
    for(int i = 0; i < C_reg_num__128/2; i++){
        C_reg0[i] = 0.0f;
        C_reg1[i] = 0.0f;
    }
    

    const int BM_idx = blockIdx.x / (cdiv(N, BLOCK_N)) * 256;
    const int BN_idx = blockIdx.x % (cdiv(N, BLOCK_N)) * 256;
    const int warp_id_m_0_1 = threadIdx.x / 64 / 4;
    const int warp_id_n_0_3 = (threadIdx.x / 64) % 4;
    const int lane_id = threadIdx.x % 64;
    A_gmem += BM_idx * K;
    B_gmem += BN_idx * K;

    for(int iter = 0; iter < total_iter_num; iter++){ 
        { // g2s
            // static_assert(total_need_transfer_num == BLOCK_M * BLOCK_K / 512 / 4);
            // static_assert(BLOCK_M * BLOCK_K % (512 * 4 * 2) == 0);
            //                                      total size                num   float  float4.
            using load_type = float2;
            constexpr int multiplier__4 = sizeof(load_type) / sizeof(__hip_bfloat16);
            constexpr int total_need_transfer_num = total_shared_size_65536 / THREAD_NUM / sizeof(load_type);
            #pragma unroll
            for(int i = 0; i < total_need_transfer_num/2; i++){
        
                const int row_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) / BLOCK_K;
                const int col_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) % BLOCK_K;
                *(load_type*)&A_smem[swizzle<4, 2, 4>(i * THREAD_NUM * multiplier__4 + threadIdx.x * multiplier__4)] = *(load_type*)&A_gmem[row_idx * K + col_idx];
            }
            
            #pragma unroll
            for(int i = 0; i < total_need_transfer_num/2; i++){
                const int row_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) / BLOCK_K;
                const int col_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) % BLOCK_K;
                *(load_type*)&B_smem[swizzle<4, 2, 4>(i * THREAD_NUM * multiplier__4 + threadIdx.x * multiplier__4)] = *(load_type*)&B_gmem[row_idx * K + col_idx];
            }
            __syncthreads();
            if(threadIdx.x == 0){
                // print_mem<16, 16>(B_smem + 10000);
            }
            __syncthreads();
        }
        
        { // s2r
            constexpr int AB_tile_reg__64 = BLOCK_M * BLOCK_K / 4 / 64;
            //                                                 分了4个块，  64个warp.
            __hip_bfloat16 B_reg[AB_tile_reg__64];
            __hip_bfloat16 A_reg0[AB_tile_reg__64];
            __hip_bfloat16 A_reg1[AB_tile_reg__64];
            // warp_M: 2 warp_N: 4
            // B_reg: 64 * 64;
            // 64 * 64 的块， 分成了 16 * 16 的 16 个块.
            // 加载时候都是加载一行连着4个从上往下.
            #pragma unroll
            for(int i = 0; i < 4; i++){ // large_row
                #pragma unroll
                for(int j = 0; j < 4; j++){ // large_col
                    *(float2*)&B_reg[i * 16 + j * 4] = 
                    *(float2*)&B_smem[
                        swizzle<4, 2, 4>(
                            (warp_id_n_0_3 * 64 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            j * 16 + lane_id / 16 * 4
                        )
                    ];
                    // for(int k = 0; k < 4; k++){ // small_col
                    //     B_reg[i * 16 + j * 4 + k] = B_smem[
                    //         (warp_id_n_0_3 * 64 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                    //         j * 16 + k + lane_id / 16 * 4
                    //     ];
                    // }
                }
            }
            #pragma unroll
            for(int i = 0; i < 4; i++){ // large_row
                #pragma unroll
                for(int j = 0; j < 4; j++){ // large_col
                    *(float2*)&A_reg0[i * 16 + j * 4] = 
                    *(float2*)&A_smem[
                        swizzle<4, 2, 4>(
                            (warp_id_m_0_1 * 128 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            j * 16 + lane_id / 16 * 4
                        )
                    ];
                    // for(int k = 0; k < 4; k++){ // small_col
                    //     A_reg0[i * 16 + j * 4 + k] = A_smem[
                    //         (warp_id_m_0_1 * 128 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                    //         j * 16 + k + lane_id / 16 * 4
                    //     ];
                    // }
                }
            }
            #pragma unroll
            for(int i = 0; i < 4; i++){ // large_row
                #pragma unroll
                for(int j = 0; j < 4; j++){ // large_col
                    *(float2*)&A_reg1[i * 16 + j * 4] = 
                    *(float2*)&A_smem[
                        swizzle<4, 2, 4>(
                            (warp_id_m_0_1 * 128 + 64 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            (j * 16 + lane_id / 16 * 4)
                        )
                    ];
                }
            }
            
            #pragma unroll
            for(int c_row = 0; c_row < 4; c_row++){
                #pragma unroll
                for(int c_col = 0; c_col < 4; c_col++){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        *((floatx4*)(C_reg0) + c_row * 4 + c_col)
                        = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                            *((float16x4*)(A_reg0) + c_row * 4 + k),
                            *((float16x4*)(B_reg) + c_col * 4 + k),
                            *((floatx4*)(C_reg0) + c_row * 4 + c_col),
                            0, 0, 0);
                    }
                }
            }
            #pragma unroll
            for(int c_row = 0; c_row < 4; c_row++){
                #pragma unroll
                for(int c_col = 0; c_col < 4; c_col++){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        *((floatx4*)(C_reg1) + c_row * 4 + c_col)
                        = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                            *((float16x4*)(A_reg1) + c_row * 4 + k),
                            *((float16x4*)(B_reg) + c_col * 4 + k),
                            *((floatx4*)(C_reg1) + c_row * 4 + c_col),
                            0, 0, 0);
                    }
                }
            }
            // mfma: 
            A_gmem += BLOCK_K;
            B_gmem += BLOCK_K;

        } // s2r
        __syncthreads();
    } // main loop for k_iter in ...
    
    // store C:
    for(int i = 0; i < 4; i++) { // row
        for(int j = 0; j < 4; j++) { // col
            for(int k = 0; k < 4; k++) { // small row
                int row = (BM_idx + i * 16 + k + (threadIdx.x & 63) / 16 * 4 + warp_id_m_0_1 * 128);
                int col = (BN_idx + j * 16 + threadIdx.x % 16) + warp_id_n_0_3 * 64;
                C_gmem[ row * N + col] = __float2bfloat16(C_reg0[i * 16 + j * 4 + k]);
            }
        }
    } 
    for(int i = 0; i < 4; i++) { // row
        for(int j = 0; j < 4; j++) { // col
            for(int k = 0; k < 4; k++) { // small row
                int row = (BM_idx + i * 16 + k + (threadIdx.x & 63) / 16 * 4 + warp_id_m_0_1 * 128 + 64);
                int col = (BN_idx + j * 16 + threadIdx.x % 16) + warp_id_n_0_3 * 64;
                C_gmem[ row * N + col] = __float2bfloat16(C_reg1[i * 16 + j * 4 + k]);
            }
        }
    } 


}