"""
FlyDSL imperative-style port of 03_fp16_gemm_gfx950_v1.hip.

Mirrors the HIP version 1:1 — same primitives, same data layout:
  - __shared__ A_smem[2][BM*BK] / B_smem[2][BN*BK]  ->  SmemAllocator + STensor (2 stages each)
  - swizzle<3,3,3>                                  ->  swizzle_xor16(row, col_bytes, k_blocks16)
  - g2s_async (buffer_load_b128 lds, voffset pre-swizzled)  ->  rocdl.raw_ptr_buffer_load_lds
  - s2r ds_read_b128                                ->  STensor.vec_load((stage, row, col), 8)
  - __builtin_amdgcn_mfma_f32_16x16x32_bf16         ->  rocdl.mfma_f32_16x16x32_bf16  (K=32 native)
  - C epilogue                                      ->  smem-restage then coalesced store
Skipped vs HIP v1 (add later): XCD remap, L2 group-M swizzle, sched_barrier hints.
"""

import functools
import sys
import torch

# kernels/ is a sibling of flydsl/ inside the FlyDSL repo — make it importable.
_FLYDSL_REPO = "/mnt/hf_hub_cache/FlyDSL"
if _FLYDSL_REPO not in sys.path:
    sys.path.insert(0, _FLYDSL_REPO)

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir.dialects import llvm, fly, memref, scf
from flydsl.compiler.protocol import fly_values

from kernels.tensor_shim import GTensor, STensor


# ============================================================
# Configuration — matches HIP 03_fp16_gemm_gfx950_v1
# ============================================================
BLOCK_M       = 256
BLOCK_N       = 256
BLOCK_K       = 64
BLOCK_M_WARPS = 2
BLOCK_N_WARPS = 4
WARP_SIZE     = 64
BLOCK_THREADS = WARP_SIZE * BLOCK_M_WARPS * BLOCK_N_WARPS    # 512
STAGES        = 2

# K=32 bf16 MFMA on gfx950: each call covers 32 K-elements with 8 bf16/lane.
WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WMMA_A_FRAG_VALUES = WMMA_M * WMMA_K // WARP_SIZE   # 8
WMMA_B_FRAG_VALUES = WMMA_N * WMMA_K // WARP_SIZE   # 8
WMMA_C_FRAG_VALUES = WMMA_M * WMMA_N // WARP_SIZE   # 4

WARP_ATOM_M = WMMA_M
WARP_ATOM_N = WMMA_N
WARP_ATOM_K = WMMA_K
WARP_M     = BLOCK_M // BLOCK_M_WARPS               # 128
WARP_N     = BLOCK_N // BLOCK_N_WARPS               # 64
WARP_M_STEPS = WARP_M // WARP_ATOM_M                # 8
WARP_N_STEPS = WARP_N // WARP_ATOM_N                # 4
WARP_K_STEPS = BLOCK_K // WARP_ATOM_K               # 2

DTYPE_BYTES = 2                                     # bf16
DMA_BYTES   = 16                                    # buffer_load_b128 lds = 16 B/lane
LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES       # 8 bf16 / lane / issue
# 4 issues per thread cover BLOCK_M*BLOCK_K = 16384 bf16 (= 256*64) for A.
LDG_REG_A_COUNT_AS = (BLOCK_M * BLOCK_K) // (LDG_ASYNC_VEC_SIZE * BLOCK_THREADS)   # 4
LDG_REG_B_COUNT_AS = (BLOCK_N * BLOCK_K) // (LDG_ASYNC_VEC_SIZE * BLOCK_THREADS)   # 4
LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE   # 8 threads cover one K-row of A
LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE   # 8 threads cover one K-row of B
BLOCK_K_LOOPS_TOTAL = None   # filled per (M,N,K)


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    """Same XOR pattern as HIP swizzle<3,3,3>: per-row 16-byte chunks get permuted."""
    return col_in_bytes ^ ((row % k_blocks16) * 16)


@functools.lru_cache(maxsize=16)
def compile_nt_gemm(m: int, n: int, k: int):
    assert m % BLOCK_M == 0 and n % BLOCK_N == 0 and k % BLOCK_K == 0
    BLOCK_K_LOOPS = k // BLOCK_K

    GPU_ARCH = get_rocm_arch()

    # ---- LDS layout: A then B, each STAGES * BLOCK_*K bf16 (= 128 KiB total). ----
    # No C-restage region — C is written directly from registers (matches HIP v1).
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES      = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_a_offset + AS_BYTES
    smem_b_offset = allocator._align(allocator.ptr, 16)
    BS_BYTES      = STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_b_offset + BS_BYTES

    KERNEL_NAME = f"nt_gemm_bf16_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_S{STAGES}_W{BLOCK_M_WARPS}x{BLOCK_N_WARPS}_K32"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def gemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        DTYPE_T = T.bf16
        acc_init  = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=DTYPE_T, shape=(m, k))
        B_ = GTensor(B, dtype=DTYPE_T, shape=(n, k))
        C_ = GTensor(C, dtype=DTYPE_T, shape=(m, n))

        from flydsl.utils.smem_allocator import SmemPtr
        base_ptr = allocator.get_base()
        sa_ptr   = SmemPtr(base_ptr, smem_a_offset, DTYPE_T, shape=(STAGES * BLOCK_M * BLOCK_K,))
        as_      = STensor(sa_ptr, DTYPE_T, shape=(STAGES, BLOCK_M, BLOCK_K))
        sb_ptr   = SmemPtr(base_ptr, smem_b_offset, DTYPE_T, shape=(STAGES * BLOCK_N * BLOCK_K,))
        bs_      = STensor(sb_ptr, DTYPE_T, shape=(STAGES, BLOCK_N, BLOCK_K))

        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x
        block_n_idx = fx.block_idx.y
        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32((BLOCK_K * DTYPE_BYTES) // 16)   # = 8 (BK=64 bytes/16B chunk)

        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M             # 0 or 128
        warp_n_idx = wid %  BLOCK_N_WARPS * WARP_N             # 0,64,128,192

        # Lane layout for the 16x16x32 MFMA s2r:
        # 16 lanes cover the 16 cols of the M/N tile; remaining 4 lane-groups handle
        # the 8 K-elements per lane (we read 8 bf16 per ds_read_b128).
        ldmatrix_a_m_idx     = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES
        ldmatrix_b_n_idx     = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES

        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS              # 2 * 8 = 16
        B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS              # 2 * 4 = 8
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS              # 8 * 4 = 32
        c_frags = [acc_init] * C_FRAGS_LEN

        # --------- async global -> LDS (replaces g2r + r2s) ---------
        def ldg_sts_a_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                global_tid  = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS_AS
                k_local_idx = global_tid %  LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                # Pre-swizzle the gmem voffset (per-thread VGPR) so the wave-coherent
                # LDS write lands in the same swizzled layout the consumer expects.
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                row_idx = m_offset + fx.Index(m_local_idx)
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)

                global_offset = A_.linear_offset((row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset    = as_.linear_offset((fx.Index(lds_stage), m_local_idx, k_local_idx)) * DTYPE_BYTES
                lds_ptr_type  = ir.Type.parse("!llvm.ptr<3>")
                lds_addr      = memref.extract_aligned_pointer_as_index(as_.memptr) + lds_offset
                lds_addr_     = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                lds_ptr       = llvm.inttoptr(lds_ptr_type, lds_addr_)

                rocdl.raw_ptr_buffer_load_lds(
                    A_.rsrc, lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )

        def ldg_sts_b_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_B_COUNT_AS):
                global_tid  = BLOCK_THREADS * i + tid
                n_local_idx = global_tid // LDG_B_X_THREADS_AS
                k_local_idx = global_tid %  LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                row_idx = n_offset + fx.Index(n_local_idx)
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)

                global_offset = B_.linear_offset((row_idx, col_idx)) * DTYPE_BYTES
                global_offset = arith.index_cast(T.i32, global_offset)
                lds_offset    = bs_.linear_offset((fx.Index(lds_stage), n_local_idx, k_local_idx)) * DTYPE_BYTES
                lds_ptr_type  = ir.Type.parse("!llvm.ptr<3>")
                lds_addr      = memref.extract_aligned_pointer_as_index(bs_.memptr) + lds_offset
                lds_addr_     = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                lds_ptr       = llvm.inttoptr(lds_ptr_type, lds_addr_)

                rocdl.raw_ptr_buffer_load_lds(
                    B_.rsrc, lds_ptr,
                    arith.constant(DMA_BYTES, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(1, type=T.i32),
                )

        # --------- s2r reads (replaces ds_read_b128 in the HIP version) ---------
        def lds_matrix_a(lds_stage):
            s = fx.Index(lds_stage)
            a_frags = [0] * (WARP_K_STEPS * WARP_M_STEPS)
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_A_FRAG_VALUES)
                    a_frags[kk * WARP_M_STEPS + ii] = vec
            return a_frags

        def lds_matrix_b(lds_stage):
            s = fx.Index(lds_stage)
            b_frags = [0] * (WARP_K_STEPS * WARP_N_STEPS)
            for ii in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_n_idx + ldmatrix_b_n_idx
                    col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = bs_.vec_load((s, row, col_in_bytes // DTYPE_BYTES), WMMA_B_FRAG_VALUES)
                    b_frags[kk * WARP_N_STEPS + ii] = vec
            return b_frags

        # --------- one BLOCK_K worth of MFMA accumulation ---------
        def block_mma_sync(a_frags, b_frags, c_frags):
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[kk * WARP_M_STEPS + ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[kk * WARP_N_STEPS + jj]
                        c_idx = ii * WARP_N_STEPS + jj
                        c_frags[c_idx] = rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(WMMA_C_FRAG_VALUES, T.f32),
                            a_frag, b_frag, c_frags[c_idx], 0, 0, 0,
                        ).res

        # ============== Prologue: load tile 0 into stage 0 ==============
        ks_begin = arith.constant(0, type=T.i32)
        ldg_sts_a_async(ks_begin, 0)
        ldg_sts_b_async(ks_begin, 0)
        rocdl.s_waitcnt(0)
        gpu.barrier()
        a_frags = lds_matrix_a(0)
        b_frags = lds_matrix_b(0)
        rocdl.sched_barrier(0)

        # ============== Main loop: iter j writes stage `next`, computes from `cur`. ==============
        init_state = [ks_begin, arith.constant(0, index=True)] + c_frags + a_frags + b_frags
        for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
            k_offset      = state[0]
            current_stage = fx.Index(state[1])
            next_stage    = 1 - current_stage
            c_frags = list(state[2 : 2 + C_FRAGS_LEN])
            a_frags = list(state[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN])
            b_frags = list(state[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN])

            # Issue async load for tile (bki) into the OTHER buffer, in parallel with compute.
            ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
            ldg_sts_b_async(k_offset + BLOCK_K, next_stage)
            block_mma_sync(a_frags, b_frags, c_frags)
            rocdl.s_waitcnt(0)
            gpu.barrier()

            a_frags_next = lds_matrix_a(next_stage)
            b_frags_next = lds_matrix_b(next_stage)
            k_offset_next = k_offset + fx.Int32(BLOCK_K)
            rocdl.sched_barrier(0)
            results = yield [k_offset_next, next_stage] + c_frags + a_frags_next + b_frags_next

        # ============== Tail: drain the last loaded tile ==============
        c_frags = list(results[2 : 2 + C_FRAGS_LEN])
        a_frags = list(results[2 + C_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN])
        b_frags = list(results[2 + C_FRAGS_LEN + A_FRAGS_LEN : 2 + C_FRAGS_LEN + A_FRAGS_LEN + B_FRAGS_LEN])
        block_mma_sync(a_frags, b_frags, c_frags)

        # ============== Epilogue: write C directly from registers. ==============
        # Per-thread MFMA C-frag layout for 16x16x32: lane (w_tid) holds
        #   col = w_tid % 16
        #   rows = (w_tid // 16) * 4 + (0..3)   [4 contiguous M rows per c_frag]
        # Each c_frag is vec4<f32> covering one (16x16) MFMA tile slice.
        c_lane_m_base = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        c_lane_n      = w_tid %  WMMA_N
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                n_global = n_offset + fx.Index(warp_atom_n_idx + c_lane_n)
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    m_global = m_offset + fx.Index(warp_atom_m_idx + c_lane_m_base + kk)
                    val = vector.extract(c_frags[ii * WARP_N_STEPS + jj],
                                         static_position=[kk], dynamic_position=[])
                    C_[m_global, n_global] = val.truncf(DTYPE_T)

    @flyc.jit
    def launch_nt_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gemm_kernel._func.__name__ = KERNEL_NAME
        gemm_kernel(A, B, C).launch(
            grid=(m // BLOCK_M, n // BLOCK_N, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_nt_gemm


def run(M: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    fn = compile_nt_gemm(M, N, K)
    fn(A, B, C, stream=fx.Stream(torch.cuda.current_stream()))
