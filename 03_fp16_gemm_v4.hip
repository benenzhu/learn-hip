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
- v3: 
-     400 TF: add XCD remap and L2 cache swizzle.  why so still so slow speed....
- v4:
-     have some register spill now? how to reduce them. 
-     first write a tile like the hipkittens do now.
-     491 TF: use float4 for g2s and s2.
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

// XCD (chiplet) configuration for MI250/MI300
#ifndef NUM_XCDS
#define NUM_XCDS 8  // MI300X has 8 XCDs
#endif

// L2 cache swizzle group size
#ifndef WGM
#define WGM 4
#endif

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



__host__ __device__ inline int chiplet_transform_chunked(
    int workgroup_id, 
    int num_workgroups,
    int num_xcds,
    int chunk_size 
) {
    // Current XCD
    int xcd = workgroup_id % num_xcds;

    // Largest full (NUM_XCDS*CHUNK_SIZE)-aligned block
    int block = num_xcds * chunk_size;
    int limit = (num_workgroups / block) * block;

    // If pid beyond the last full block, leave unchanged
    if (workgroup_id > limit) return workgroup_id;

    // Local PID (within round-robin assignment)
    int local_pid    = workgroup_id / num_xcds;
    int chunk_idx    = local_pid / chunk_size;
    int pos_in_chunk = local_pid % chunk_size;

    // New PID
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}




// template <int row=256, int col=64, int BB = 4, int MM = 2, int SS = 4>
// struct smem_tile{
//     __hip_bfloat16 data[row * col]; 
//     static constexpr int _row = row;
//     static constexpr int _col = col;
// };


// struct gmem_tile{
//     __hip_bfloat16 *ptr;
//     const int row;
//     const int col;
//     gmem_tile(int row, int col, __hip_bfloat16 *ptr): row(row), col(col), ptr(ptr){
//     }
// };


// template <int float_size=2>
// void g2s(smem_tile<256, 64> &s_tile, gmem_tile &g_tile, const int g_row, const int g_col){
//     if constexpr(float_size==2){
//         constexpr int multiplier__4 = sizeof(float2) / sizeof(__hip_bfloat16);
//         using s_tile_type = std::remove_reference_t<decltype(s_tile)>;
//         static constexpr int total_need_transfer_num = s_tile_type::_col * s_tile_type::_row / multiplier__4;
//         #pragma unroll
//         for(int i = 0; i < total_need_transfer_num/2; i++){
//             const int  
//         }
//         for(int i = 0; i < 4; i++) {
//             #pragma unroll
//             for(int j = 0; j < 4; j++) {
//                 s_tile.data[g_row + i * 16 + j * 4] = g_tile.ptr[g_row + i * 16 + j * 4];
//             }
//         }
//     }
// }




__launch_bounds__(THREAD_NUM)
__global__
void _3_fp16_gemm_v4(
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
    
    constexpr int C_reg_half__64 = BLOCK_M * BLOCK_N / 512 / 2;
    constexpr int warp_reg_num__8192 = C_reg_half__64 * 2 * 64;
    float C_reg0[C_reg_half__64];
    float C_reg1[C_reg_half__64];
    for(int i = 0; i < C_reg_half__64; i++){
        C_reg0[i] = 0.0f;
        C_reg1[i] = 0.0f;
    }
    
    // ========== XCD Remap + L2 Cache Swizzle ==========
    // Step 1: Get total workgroup id (supports both 1D and 2D grids)
    int wgid = blockIdx.x + blockIdx.y * gridDim.x;
    const int NUM_WGS = gridDim.x * (gridDim.y > 0 ? gridDim.y : 1);
    
    // Step 2: XCD remap - distribute workgroups to same XCD in chunks
    // This reduces cross-chiplet communication on MI250/MI300
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    
    // Step 3: L2 Cache Swizzle with Group M pattern
    // Groups WGM consecutive M-tiles together for better L2 locality
    const int num_pid_m = cdiv(M, BLOCK_M);
    const int num_pid_n = cdiv(N, BLOCK_N);
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m = min(num_pid_m - first_pid_m, WGM);
    const int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    const int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // if(threadIdx.x == 0){
    //     printf("xcd: %02d, pid_m: %02d, pid_n: %d, blockIdx.x: %d\n",  blockIdx.x % 8, pid_m, pid_n, blockIdx.x);
    // }
    // return;
    
    // Step 4: Convert tile indices to actual coordinates
    const int BM_idx = pid_m * BLOCK_M;
    const int BN_idx = pid_n * BLOCK_N;
    const int warp_id_m_0_1 = threadIdx.x / 64 / 4;
    const int warp_id_n_0_3 = (threadIdx.x / 64) % 4;
    const int lane_id = threadIdx.x % 64;
    A_gmem += BM_idx * K;
    B_gmem += BN_idx * K;


    constexpr int total_iter_num = cdiv(K, BLOCK_K);
    for(int iter = 0; iter < total_iter_num; iter++){   
        { // first tile g2s...
            // static_assert(total_need_transfer_num == BLOCK_M * BLOCK_K / 512 / 4);
            // static_assert(BLOCK_M * BLOCK_K % (512 * 4 * 2) == 0);
            //                                      total size                num   float  float4.
            using load_type = float4;
            constexpr int multiplier__4 = sizeof(load_type) / sizeof(__hip_bfloat16);
            constexpr int total_need_transfer_num = total_shared_size_65536 / THREAD_NUM / sizeof(load_type);
            #pragma unroll
            for(int i = 0; i < total_need_transfer_num/2; i++) {
                const int row_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) / BLOCK_K;
                const int col_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) % BLOCK_K;
                *(load_type*)&A_smem[swizzle<3, 3, 3>(i * THREAD_NUM * multiplier__4 + threadIdx.x * multiplier__4)] = *(load_type*)&A_gmem[row_idx * K + col_idx];
            }
            
            #pragma unroll
            for(int i = 0; i < total_need_transfer_num/2; i++){
                const int row_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) / BLOCK_K;
                const int col_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) % BLOCK_K;
                *(load_type*)&B_smem[swizzle<3, 3, 3>(i * THREAD_NUM * multiplier__4 + threadIdx.x * multiplier__4)] = *(load_type*)&B_gmem[row_idx * K + col_idx];
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
                for(int j = 0; j < 2; j++){ // large_col
                    *(float4*)&B_reg[i * 16 + j * 8] = 
                    *(float4*)&B_smem[
                        swizzle<3, 3, 3>(
                            (warp_id_n_0_3 * 64 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            j * 32 + lane_id / 16 * 8
                        )
                    ];
                }
            }
            #pragma unroll
            for(int i = 0; i < 4; i++){ // large_row
                #pragma unroll
                for(int j = 0; j < 2; j++){ // large_col
                    *(float4*)&A_reg0[i * 16 + j * 8] = 
                    *(float4*)&A_smem[
                        swizzle<3, 3, 3>(
                            (warp_id_m_0_1 * 128 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            j * 32 + lane_id / 16 * 8
                        )
                    ];
                }
            }
            #pragma unroll
            for(int i = 0; i < 4; i++){ // large_row
                #pragma unroll
                for(int j = 0; j < 2; j++){ // large_col
                    *(float4*)&A_reg1[i * 16 + j * 8] = 
                    *(float4*)&A_smem[
                        swizzle<3, 3, 3>(
                            (warp_id_m_0_1 * 128 + 64 + i * 16 + threadIdx.x % 16) * BLOCK_K +
                            (j * 32 + lane_id / 16 * 8)
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