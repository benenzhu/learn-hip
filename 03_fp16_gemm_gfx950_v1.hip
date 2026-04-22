// empty lines for hiprtc.















/*
- 03_v0: 
    - use 512 thread
    - simple load and calculate.
    - pre register allocate.
    - 204 VGPRs;
    - Should have bank conflict problem and load problems.
- [x] use float4 load
- v0: 137 TF:
- v1: 136 TF: use float2 load for shared memory, 
    - not up. so prob mainly for the bank conflict problem?
- v2: 
    - 165 TF:  remove clear shared memory.
    - 162 TF: use float2 load for shared memory.
    - 382 TF: add swizzle for shared memory.  (+136%)
    - 396 TF: add all unroll for params.
- v3: 
    - 400 TF: add XCD remap and L2 cache swizzle.  why so still so slow speed....
- v4:
    - have some register spill now? how to reduce them. 
    - first write a tile like the hipkittens do now.
    - 491 TF: use float4 for g2s and s2.
- v5:
    - 497 TF: add fast float2bfloat16... now ignore... maybe put into last.
    - 602 TF: for close barrier...
    - 535 TF: for open barrier... ping pong should have much more effect. learn it more.
    - so shared memory interleave can do much more... 
- v6: 
    - simple refact. to reduce register pressure and used for later optimizations.

- [ ] merge A_smem & B_smem
- [ ] small tile mma, add 8 wave ping-pong interleave.
*/







#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <bit>
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

// __device__ float fast_float2bfloat16(float value){
//     return std::bit_cast<__hip_bfloat16>(
//         static_cast<__hip_internal::uint16_t>(
//             std::bit_cast<__hip_internal::uint32_t>(value) >> 16
//         )
//     );        
// };



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


__device__ void g2s(__hip_bfloat16 *shared_mem, __hip_bfloat16 *gmem) { // first tile g2s...
    using load_type = float4;
    constexpr int multiplier__4 = sizeof(load_type) / sizeof(__hip_bfloat16);
    constexpr int total_shared_size_65536 = 65536;
    constexpr int total_need_transfer_num = total_shared_size_65536 / THREAD_NUM / sizeof(load_type) / 2;
    // #pragma unroll
    for(int i = 0; i < total_need_transfer_num; i++) {
        const int row_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) / BLOCK_K;
        const int col_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__4) % BLOCK_K;
        *(load_type*)&shared_mem[swizzle<3, 3, 3>(i * THREAD_NUM * multiplier__4 + threadIdx.x * multiplier__4)] = *(load_type*)&gmem[row_idx * K + col_idx];
    }
};

using uint32_t =__hip_internal::uint32_t;
using uint16_t = __hip_internal::uint16_t;
using uint8_t = __hip_internal::uint8_t;
using uint64_t = __hip_internal::uint64_t;

using i32x4 = uint32_t __attribute__((ext_vector_type(4)));
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}

__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes, uint32_t row_stride_bytes = 0) {
    uint64_t as_int = reinterpret_cast<uint64_t>(ptr);   // width = sizeof(void*)
    uint64_t  as_u64 = static_cast<uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, 0x110000);

    row_stride_bytes &= 0x3FFF;
    if (row_stride_bytes) {
        // - The swizzle stride lives in bits 13:0 of word2.
        //   Max value = 0x3FFF (8 KiB – one cache line per bank).
        uint64_t stride_field = row_stride_bytes;
        stride_field = stride_field | 0x4000;         // Cache swizzle
        stride_field = stride_field | 0x8000;         // Swizzle enable
        rsrc.ptr |= stride_field << 48;
    }

    return *reinterpret_cast<const i32x4*>(&rsrc);
}

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b8(uint8_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i8");

__device__ void llvm_amdgcn_raw_buffer_store_b16(uint16_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ void llvm_amdgcn_raw_buffer_store_b32(uint32_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b64(uint64_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");

// Direct global -> LDS buffer load (gfx940+; gfx950 supports widths 4 / 12 / 16 B).
// Per-lane payload of `size` bytes lands at LDS[ m0 + lane * size ], where m0 is set
// from `lds_ptr` (must be wave-uniform). Each lane provides its own `voffset`.
// We exploit this to put the swizzle on the gmem-side voffset (per-lane VGPR) so the
// natural per-lane LDS layout matches the same swizzled layout produced by the old
// g2r + r2s pair — `s2r` and the pre-computed `*_smem_load_base` arrays don't change.
__device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 srsrc,
    __attribute__((address_space(3))) uint32_t* lds_ptr,
    uint32_t size,
    uint32_t voffset,
    uint32_t soffset,
    uint32_t offset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

// Async global -> LDS for one BLOCK_M x BLOCK_K (or BLOCK_N x BLOCK_K) tile.
// Replaces the old g2r(...) + r2s(...) pair, eliminating A_buff/B_buff VGPR storage.
// Completion is tracked by vmcnt; caller does s_waitcnt vmcnt(0) + __syncthreads
// before consuming the loaded tile.
__device__ __forceinline__ void g2s_async(__hip_bfloat16 *shared_mem, __hip_bfloat16 *gmem) {
    constexpr int load_bytes    = 16;                       // gfx950 buffer_load_b128 lds
    constexpr int multiplier__8 = sizeof(float4) / sizeof(__hip_bfloat16);  // 8 bf16/lane
    constexpr int total_bf16    = BLOCK_M * BLOCK_K;        // 16384 (32 KiB)
    constexpr int total_issues  = total_bf16 / THREAD_NUM / multiplier__8;  // 4

    const int row_stride_bytes = K * sizeof(__hip_bfloat16);
    const int total_bytes      = K * 256 * sizeof(__hip_bfloat16);
    i32x4 srsrc = make_srsrc(gmem, total_bytes, row_stride_bytes);

    // wave_id forced into scalar reg so the lds-pointer arithmetic stays uniform.
    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);

    #pragma unroll
    for (int i = 0; i < total_issues; i++) {
        // This thread's natural LDS chunk (in bf16 element index, before swizzle).
        const int natural_lds_elt = i * THREAD_NUM * multiplier__8 + threadIdx.x * multiplier__8;
        // Pre-swizzle to determine which gmem element this thread must fetch so that
        // the wave-coherent LDS write reproduces the swizzled layout.
        const int gmem_elt = swizzle<3,3,3>(natural_lds_elt);
        const int row_idx  = gmem_elt / BLOCK_K;
        const int col_idx  = gmem_elt % BLOCK_K;
        const uint32_t voffset = (row_idx * K + col_idx) * sizeof(__hip_bfloat16);

        // Wave-uniform LDS base for this issue. Hardware adds lane * 16 B per lane.
        __hip_bfloat16* lds_wave_base =
            shared_mem + i * THREAD_NUM * multiplier__8 + wave_id * 64 * multiplier__8;

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc,
            (__attribute__((address_space(3))) uint32_t*)(uint64_t)lds_wave_base,
            load_bytes,
            voffset,
            0, 0, 0);
    }
}

__device__ void g2r(__hip_bfloat16 *reg, __hip_bfloat16 *gmem) { // first tile g2s...
    using load_type = float4;
    constexpr int multiplier__8 = sizeof(load_type) / sizeof(__hip_bfloat16);
    constexpr int total_shared_size_65536 = 65536;
    constexpr int total_need_transfer_num = total_shared_size_65536 / THREAD_NUM / sizeof(load_type) / 2;
    
    const int row_stride = K;
    const int row_stride_bytes = row_stride * sizeof(__hip_bfloat16);
  
    // buffer resource
    const int total_bytes = row_stride * 256 * sizeof(__hip_bfloat16);
    i32x4 srsrc = make_srsrc(gmem, total_bytes, row_stride_bytes);

    const int laneid = threadIdx.x % 512;

    // #pragma unroll
    for(int i = 0; i < total_need_transfer_num; i++) {
        const int row_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__8) / BLOCK_K;
        const int col_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__8) % BLOCK_K;
        const int byte_offset = (row_idx * K + col_idx) * sizeof(__hip_bfloat16);
        __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
        *(load_type*)&reg[i * multiplier__8] = *reinterpret_cast<load_type*>(&raw);
    }
};

__device__ void r2s(__hip_bfloat16 *shared, __hip_bfloat16 *reg) {
    using load_type = float4;
    constexpr int multiplier__8 = sizeof(load_type) / sizeof(__hip_bfloat16);
    constexpr int total_shared_size_65536 = 65536;
    constexpr int total_need_transfer_num = total_shared_size_65536 / THREAD_NUM / sizeof(load_type) / 2;
    // #pragma unroll
    for(int i = 0; i < total_need_transfer_num; i++) {
        const int row_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__8) / BLOCK_K;
        const int col_idx = ((THREAD_NUM * i + threadIdx.x) * multiplier__8) % BLOCK_K;
        *(load_type*)&shared[swizzle<3, 3, 3>(i * THREAD_NUM * multiplier__8 + threadIdx.x * multiplier__8)] = *(load_type*)&reg[i * multiplier__8];
    }
}

template<int large_row_inc, int j, int half_which>
__device__ void s2r(__hip_bfloat16 *reg, __hip_bfloat16 *shared){
    // 256 * 64
    #pragma unroll
    for(int i = half_which * 2; i < 2 * half_which + 2; i++){
        // asm volatile(
        //     "ds_read_b128 %0, %1, offset: %2\n"
        //     : "=v"(*reinterpret_cast<float4*>(reg + i * 8))
        //     : "v"(addr), "i"(i * 16 * BLOCK_K * 2)
        //     : "memory"
        // );
        *(float4*)&reg[i * 8] = *(float4*)&shared[i * 16 * BLOCK_K + large_row_inc * 64 * BLOCK_K];
    }
    // TODO: optimize this
    // asm volatile("s_waitcnt lgkmcnt(0)");
}


template <int HALF_ROW>
__device__ void mma(float *C_reg, __hip_bfloat16 A_reg[32], __hip_bfloat16 B_reg[32]){
    // gfx950 (MI355X) K=32 bf16 MFMA — one call replaces the prior K=16 _1k pair.
    // Operand layout per lane: float16x8 (8 bf16). The existing s2r packing already
    // supplies 8 bf16/lane via two ds_read_b128 (s2r<...,half=0> + s2r<...,half=1>),
    // so A_reg / B_reg layout is unchanged — only the inner k-loop collapses.
    // 32-element register layout now: 4 c_row * 8 elements = 32 (instead of 4*2*4).
    using float16x8 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using floatx4   = __attribute__((__vector_size__(4 * sizeof(float))))    float;
    static_assert(3 * 8 + 7 < 32);  // max bf16 index < 32
    #pragma unroll
    for(int c_row = HALF_ROW * 2; c_row < 2 * HALF_ROW + 2; c_row++){
        #pragma unroll
        for(int c_col = 0; c_col < 4; c_col++){
            *((floatx4*)(C_reg) + c_row * 4 + c_col)
            = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                *((float16x8*)(A_reg) + c_row),   // 8 bf16/lane = K=32 in one MFMA
                *((float16x8*)(B_reg) + c_col),
                *((floatx4*)(C_reg) + c_row * 4 + c_col),
                0, 0, 0);
        }
    }
}


#define BARRIER(prio) __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0); __builtin_amdgcn_s_setprio(prio);
#define BARRIER(prio)


// ----------------------------------------------------------------------------
// compute_tile<BUF>: pure s2r + mma over one BLOCK_K. No global prefetch — the
// async global->LDS load for the *next* tile is issued from the main loop, in
// parallel with this body, and tracked via vmcnt.
// BUF is a constexpr buffer index into the pre-computed swizzled base-pointer
// arrays; ds_read_b128 stays a constant-offset load.
// ----------------------------------------------------------------------------
template <int BUF>
__device__ __forceinline__ void compute_tile(
    float* C_reg0, float* C_reg1,
    __hip_bfloat16* A_reg, __hip_bfloat16* A_reg2, __hip_bfloat16* B_reg,
    __hip_bfloat16* const A_ld [2], __hip_bfloat16* const A_ld2[2],
    __hip_bfloat16* const B_ld [2], __hip_bfloat16* const B_ld2[2])
{
    s2r<0,0,0>(B_reg, B_ld[BUF]);
    s2r<0,0,1>(B_reg, B_ld[BUF]);
    s2r<0,0,0>(A_reg, A_ld[BUF]);
    BARRIER(1);

    mma<0>(C_reg0, A_reg, B_reg);

    s2r<0,0,1>(A_reg, A_ld[BUF]);
    mma<1>(C_reg0, A_reg, B_reg);
    BARRIER(0);

    s2r<1,0,0>(A_reg2, A_ld[BUF]);
    BARRIER(1);
    s2r<1,0,1>(A_reg2, A_ld[BUF]);
    mma<0>(C_reg1, A_reg2, B_reg);
    mma<1>(C_reg1, A_reg2, B_reg);
    BARRIER(0);

    s2r<0,1,0>(B_reg, B_ld2[BUF]);
    s2r<0,1,1>(B_reg, B_ld2[BUF]);
    s2r<0,1,0>(A_reg, A_ld2[BUF]);

    BARRIER(1);
    mma<0>(C_reg0, A_reg, B_reg);
    s2r<0,1,1>(A_reg, A_ld2[BUF]);
    mma<1>(C_reg0, A_reg, B_reg);

    BARRIER(0);

    s2r<1,1,0>(A_reg2, A_ld2[BUF]);
    BARRIER(1);
    mma<0>(C_reg1, A_reg2, B_reg);
    s2r<1,1,1>(A_reg2, A_ld2[BUF]);
    mma<1>(C_reg1, A_reg2, B_reg);
    BARRIER(0);
}

// __attribute__((amdgpu_flat_work_group_size(512, 512)))
// __attribute__((amdgpu_num_vgpr(128)))  // 限制VGPR到128，强制编译器用AGPR存累加器
__launch_bounds__(THREAD_NUM, 2)
__global__
void _3_fp16_gemm_gfx950_v1(
  __hip_bfloat16* __restrict__ A_gmem,
  __hip_bfloat16* __restrict__ B_gmem,
  __hip_bfloat16* __restrict__ C_gmem
) {
    // __hip_bfloat16 Atile[]
    
    const int tid = threadIdx.x;
    
    // g2s
    // s2r
    // doing calculate
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    
    // Double-buffered SMEM for gfx950: total 131,072 B (fits in ~160 KB/CU LDS).
    // Declared as [2][...] (not flat [2 * STRIDE]) so the two halves are distinct
    // sub-objects — LLVM can prove producer r2s into A_smem[1] is alias-independent
    // from consumer s2r out of A_smem[0], enabling reordering across the inner loop.
    __shared__ __hip_bfloat16 A_smem[2][BLOCK_M * BLOCK_K];
    __shared__ __hip_bfloat16 B_smem[2][BLOCK_N * BLOCK_K];
    constexpr int total_shared_size_131072 = sizeof(A_smem) + sizeof(B_smem);
    
    constexpr int C_reg_half__64 = BLOCK_M * BLOCK_N / 512 / 2;
    constexpr int warp_reg_num__8192 = C_reg_half__64 * 2 * 64;
    float C_reg0[C_reg_half__64];
    float C_reg1[C_reg_half__64];
    constexpr int AB_tile_reg__32 = 64 * 32 / 64;
    __hip_bfloat16 B_reg[AB_tile_reg__32];
    __hip_bfloat16 A_reg[AB_tile_reg__32];
    __hip_bfloat16 A_reg2[AB_tile_reg__32];
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

    // Prologue: async-load tile 0 directly into LDS buf 0 (no register staging).
    g2s_async(A_smem[0], A_gmem);
    g2s_async(B_smem[0], B_gmem);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    const int warp_row = __builtin_amdgcn_readfirstlane(threadIdx.x / 64 / 4);
    if(warp_row == 1){
        BARRIER(0);
    }

    // Pre-computed swizzled base pointers, one per buffer. Each `compute_tile<BUF>`
    // selects with a constexpr index so ds_read_b128 keeps a constant offset
    // (no swizzle XOR in the inner loop).
    const int A_swz_lo = swizzle<3,3,3>((warp_id_m_0_1 * 2 * 64 + threadIdx.x % 16) * BLOCK_K + lane_id / 16 * 8);
    const int A_swz_hi = swizzle<3,3,3>((warp_id_m_0_1 * 2 * 64 + threadIdx.x % 16) * BLOCK_K + lane_id / 16 * 8 + 32);
    const int B_swz_lo = swizzle<3,3,3>((warp_id_n_0_3      * 64 + threadIdx.x % 16) * BLOCK_K + lane_id / 16 * 8);
    const int B_swz_hi = swizzle<3,3,3>((warp_id_n_0_3      * 64 + threadIdx.x % 16) * BLOCK_K + lane_id / 16 * 8 + 32);

    __hip_bfloat16* A_smem_load_base [2] = { A_smem[0] + A_swz_lo, A_smem[1] + A_swz_lo };
    __hip_bfloat16* A_smem_load_base2[2] = { A_smem[0] + A_swz_hi, A_smem[1] + A_swz_hi };
    __hip_bfloat16* B_smem_load_base [2] = { B_smem[0] + B_swz_lo, B_smem[1] + B_swz_lo };
    __hip_bfloat16* B_smem_load_base2[2] = { B_smem[0] + B_swz_hi, B_smem[1] + B_swz_hi };
    
    // Double-buffered main loop with direct global->LDS async copy.
    // Pattern (per pair of tiles):
    //   issue async load tile (2*iter+1) -> buf 1   (in flight, vmcnt-tracked)
    //   compute_tile<0> from buf 0                  (overlaps with the async load)
    //   waitcnt + barrier
    //   issue async load tile (2*iter+2) -> buf 0
    //   compute_tile<1> from buf 1                  (overlaps with the async load)
    //   waitcnt + barrier
    // No A_buff / B_buff register staging — the spill goes away.
    constexpr int total_iter_num = cdiv(K, BLOCK_K);
    static_assert(total_iter_num >= 2 && total_iter_num % 2 == 0,
                  "loop unroll-by-2 requires an even tile count");
    constexpr int loop_iters = (total_iter_num - 2) / 2;   // last 2 tiles handled in tail

    for (int iter = 0; iter < loop_iters; iter++) {
        // ---- tile (2*iter): compute from buf 0, async-prefetch next into buf 1 ----
        A_gmem += BLOCK_K;
        B_gmem += BLOCK_K;
        g2s_async(A_smem[1], A_gmem);
        g2s_async(B_smem[1], B_gmem);
        compute_tile<0>(C_reg0, C_reg1, A_reg, A_reg2, B_reg,
                        A_smem_load_base, A_smem_load_base2,
                        B_smem_load_base, B_smem_load_base2);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();

        // ---- tile (2*iter+1): compute from buf 1, async-prefetch next into buf 0 ----
        A_gmem += BLOCK_K;
        B_gmem += BLOCK_K;
        g2s_async(A_smem[0], A_gmem);
        g2s_async(B_smem[0], B_gmem);
        compute_tile<1>(C_reg0, C_reg1, A_reg, A_reg2, B_reg,
                        A_smem_load_base, A_smem_load_base2,
                        B_smem_load_base, B_smem_load_base2);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
    } // main loop

    if (warp_row == 0) {
        BARRIER(0);
    }

    // ---- Tail: last 2 tiles (indices total-2 and total-1) ----
    // Tile total-2 sits in buf 0; async-prefetch tile total-1 into buf 1 while
    // computing total-2, then drain tile total-1 from buf 1 with no further prefetch.
    {
        A_gmem += BLOCK_K;
        B_gmem += BLOCK_K;
        g2s_async(A_smem[1], A_gmem);
        g2s_async(B_smem[1], B_gmem);
        compute_tile<0>(C_reg0, C_reg1, A_reg, A_reg2, B_reg,
                        A_smem_load_base, A_smem_load_base2,
                        B_smem_load_base, B_smem_load_base2);
        // Stop the scheduler from hoisting compute_tile<1>'s ds_read_b128s into
        // compute_tile<0>'s tail MFMAs — that's what was stretching one C-slot's
        // live range across the boundary and forcing the spill.
        // __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        // __builtin_amdgcn_sched_barrier(0);

        compute_tile<1>(C_reg0, C_reg1, A_reg, A_reg2, B_reg,
                        A_smem_load_base, A_smem_load_base2,
                        B_smem_load_base, B_smem_load_base2);
    }
    // asm volatile("" ::: "memory");  // 编译器屏障，阻止重排
    // asm volatile("s_waitcnt lgkmcnt(0)");
    // asm volatile("s_waitcnt vmcnt(0)");
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);  
    
    // store C:
    {
        #pragma unroll
        for(int i = 0; i < 4; i++) { // row
            #pragma unroll
            for(int j = 0; j < 4; j++) { // col
                #pragma unroll
                for(int k = 0; k < 4; k++) { // small row
                    const int row = (BM_idx + i * 16 + k + (threadIdx.x & 63) / 16 * 4 + warp_id_m_0_1 * 128);
                    const int col = (BN_idx + j * 16 + threadIdx.x % 16) + warp_id_n_0_3 * 64;
                    C_gmem[ row * N + col] = __float2bfloat16(C_reg0[i * 16 + j * 4 + k]);
                }
            }
        } 
        #pragma unroll
        for(int i = 0; i < 4; i++) { // row
            #pragma unroll
            for(int j = 0; j < 4; j++) { // col
                #pragma unroll
                for(int k = 0; k < 4; k++) { // small row
                    const int row = (BM_idx + i * 16 + k + (threadIdx.x & 63) / 16 * 4 + warp_id_m_0_1 * 128 + 64);
                    const int col = (BN_idx + j * 16 + threadIdx.x % 16) + warp_id_n_0_3 * 64;
                    C_gmem[ row * N + col] = __float2bfloat16(C_reg1[i * 16 + j * 4 + k]);
                }
            }
        } 
    }
}