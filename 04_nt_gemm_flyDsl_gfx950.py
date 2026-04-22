"""
FlyDSL port of 03_fp16_gemm_gfx950_v1.hip — NT bf16 GEMM for MI355X (gfx950).

Layout / structure mirrors examples/04-preshuffle_gemm.py from the FlyDSL repo:
  - composed swizzle layout (3,3,3) for A in LDS, double-buffered (STAGES_A = 2)
  - run_pipeline_stage(read_stage, next_k) helper, called twice per loop iter
    (constexpr stage 0/1 — same unroll-by-2 trick as the HIP version)
  - hot_loop_scheduler() with sched_dsrd / sched_mfma / sched_vmem hints

Differences vs example 04:
  - bf16 instead of f16
  - Sized to match HIP v1: BLOCK_M=BLOCK_N=256, BLOCK_K=64, NUM_WARP_M=2, NUM_WARP_N=4
  - gfx950 K=32 MFMA: MFMA(16, 16, 32, BFloat16)
  - No preshuffle on B (direct buffer copy of B from gmem each iter)
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


# --- Tile / pipeline config (matches HIP 03_fp16_gemm_gfx950_v1) ---
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
STAGES_A = 2          # double-buffered LDS for A
NUM_WARP_M = 2
NUM_WARP_N = 4
NUM_THREADS = 64 * NUM_WARP_M * NUM_WARP_N    # 512

# MFMA shape. The high-level fx.rocdl.MFMA atom only registers K=16 bf16 today
# (CDNA3 atoms — see lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp::mfma_f32_16x16x16bf16_1k).
# Switching to K=32 (gfx950 native) requires direct rocdl.mfma_f32_16x16x32_bf16
# call instead of the atom — leave for v1.
MFMA_M, MFMA_N, MFMA_K = 16, 16, 16

# LDS bytes: only A is staged in LDS (B goes g→r direct, like example 04).
SMEM_BYTES = STAGES_A * BLOCK_M * BLOCK_K * 2  # bf16 = 2 B/elt


@functools.lru_cache(maxsize=16)
def compile_nt_gemm(M: int, N: int, K: int):
    """Build & cache the JIT'd kernel for one (M, N, K) shape.
    K is baked in as a Python constexpr (it controls the unrolled k-loop bounds)."""

    assert M  % BLOCK_M == 0
    assert N  % BLOCK_N == 0
    assert K  % BLOCK_K == 0
    assert (K // BLOCK_K) % 2 == 0, "loop unroll-by-2 needs even tile count"

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def gemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        # NT layout: A is (M, K) row-major, B is (N, K) row-major, C = A @ B^T.
        gA_k = fx.flat_divide(A, fx.make_tile(BLOCK_M, BLOCK_K))[None, None, bid_x, None]   # (BM, BK, k)
        gB_k = fx.flat_divide(B, fx.make_tile(BLOCK_N, BLOCK_K))[None, None, bid_y, None]   # (BN, BK, k)
        gC   = fx.flat_divide(C, fx.make_tile(BLOCK_M, BLOCK_N))[None, None, bid_x, bid_y]  # (BM, BN)

        thr_mma         = tiled_mma.thr_slice(tid)
        thr_copy_g2s_A  = tiled_copy_g2s_A.get_slice(tid)

        uni_copy_128b    = fx.make_copy_atom(fx.UniversalCopy128b(),    fx.BFloat16)
        buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        # B-fragment retile yields 2 non-contiguous 8-byte halves with K=16 MFMA
        # and no K-permutation — drop to b64 (one contiguous 4-bf16 chunk per issue).
        buffer_copy_64b  = fx.make_copy_atom(fx.rocdl.BufferCopy64b(),  fx.BFloat16)
        # C r→g atom: bf16 element is 2 bytes, so BufferCopy16b is 1 element/issue.
        # Wider atoms need contiguous mma_frag_C tiles which depends on mma layout.
        buffer_copy_16b  = fx.make_copy_atom(fx.rocdl.BufferCopy16b(),  fx.BFloat16)
        buffer_copy_32b  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(),  fx.BFloat16)

        thr_copy_s2r_A = fx.make_tiled_copy_A(buffer_copy_128b, tiled_mma).get_slice(tid)
        thr_copy_g2r_B = fx.make_tiled_copy_B(buffer_copy_64b,  tiled_mma).get_slice(tid)
        thr_copy_r2g_C = fx.make_tiled_copy_C(buffer_copy_16b,  tiled_mma).get_slice(tid)

        # Shared memory: A only (B doesn't go through LDS in this v0 port).
        smem_ptr      = fx.get_dyn_shared()
        smem_ptr_bf16 = fx.recast_iter(
            fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 512), smem_ptr
        )

        # SwizzleType.get(3,3,3) is the same XOR pattern as the HIP swizzle<3,3,3>:
        # XOR bit[5:3] with bit[8:6], 8-element (16-byte) coherent chunks.
        composed_layout_A = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 3, 3)),
            fx.make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A), (1, 0, 2)),
        )
        sA = fx.make_view(smem_ptr_bf16, composed_layout_A)   # (BM, BK, STAGES_A)

        thr_gA_k    = thr_copy_g2s_A.partition_S(gA_k)        # (VA, VM, VK, k)
        thr_sA      = thr_copy_g2s_A.partition_D(sA)          # (VA, VM, VK, STAGES_A)
        thr_sA_s2r  = thr_copy_s2r_A.partition_S(sA)          # (VA, VM, VK, STAGES_A)
        thr_gB_k    = thr_copy_g2r_B.partition_S(gB_k)        # (VB, VN, VK, k)
        thr_gC      = thr_copy_r2g_C.partition_S(gC)          # (VC, VM, VN)

        copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])  # g→r staging for A

        mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
        mma_frag_B = fx.make_fragment_like(
            fx.flat_product(
                thr_mma.partition_B(gB_k).layout(None, None, None, 0),
                fx.make_layout(2, 1),    # 2 stages held in registers for B
            ),
            fx.BFloat16.ir_type,
        )
        mma_frag_C = thr_mma.make_fragment_C(gC)

        mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
        mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

        # Accumulator is f32; cast to bf16 before storing.
        mma_frag_C_bf16   = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
        mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_bf16)

        def run_pipeline_stage(read_stage, next_k, read_next=True):
            write_stage = read_stage ^ 1

            if read_next:
                next_k_i = fx.Int32(next_k)
                # A: g→r into copy_frag_A (simple per-tile addressing).
                fx.copy(
                    buffer_copy_128b,
                    thr_gA_k[None, None, None, next_k_i],
                    copy_frag_A,
                )
                # B: g→r straight into the MMA B-fragment for write_stage (b64 atom).
                fx.copy(
                    buffer_copy_64b,
                    thr_gB_k[None, None, None, next_k_i],
                    mma_frag_B_retile[None, None, None, write_stage],
                )

            # K-loop. With MFMA_K=32 and BLOCK_K=64 → 2 inner k-steps per tile.
            for block_k_iter in fx.range_constexpr(BLOCK_K // MFMA_K):
                fx.copy(
                    uni_copy_128b,
                    thr_sA_s2r[None, None, block_k_iter, read_stage],
                    mma_frag_A_retile[None, None, block_k_iter],
                )
                fx.gemm(
                    tiled_mma,
                    mma_frag_C,
                    mma_frag_A[None, None, block_k_iter],
                    mma_frag_B[None, None, block_k_iter, read_stage],
                    mma_frag_C,
                )

            # r→s: stage the just-loaded A into LDS write_stage.
            fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, write_stage])
            fx.gpu.barrier()

            # Inline scheduler hints (mirrors example 04).
            def hot_loop_scheduler():
                fx.rocdl.sched_dsrd(2)
                fx.rocdl.sched_mfma(2)
                fx.rocdl.sched_dsrd(1)
                fx.rocdl.sched_mfma(1)
                fx.rocdl.sched_dsrd(1)
                fx.rocdl.sched_mfma(2)

                def sched_main_iter(with_vmem=False, with_dswr=False):
                    if with_vmem:
                        fx.rocdl.sched_vmem(1)
                    fx.rocdl.sched_mfma(2)
                    fx.rocdl.sched_dsrd(1)
                    fx.rocdl.sched_mfma(2)
                    if with_dswr:
                        fx.rocdl.sched_dswr(1)

                for _ in fx.range_constexpr(8):
                    sched_main_iter(with_vmem=True)
                sched_main_iter()
                for _ in fx.range_constexpr(7):
                    sched_main_iter(with_dswr=True)

                fx.rocdl.sched_barrier(0)

            hot_loop_scheduler()

        # ---- Prologue: tile 0 of A into stage 0, tile 0 of B into reg stage 0. ----
        fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, 0], copy_frag_A)
        fx.copy(buffer_copy_64b,  thr_gB_k[None, None, None, 0],
                mma_frag_B_retile[None, None, None, 0])

        # Per-thread C accumulator: BLOCK_M*BLOCK_N / NUM_THREADS = 256*256/512 = 128 floats.
        C_PER_THREAD = BLOCK_M * BLOCK_N // NUM_THREADS
        mma_frag_C.store(fx.arith.constant_vector(0.0, fx.T.VectorType.get([C_PER_THREAD], fx.T.f32())))

        fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
        fx.gpu.barrier()

        # ---- Main loop, unrolled by 2 stages so read_stage is constexpr. ----
        for k_iter in range(0, K // BLOCK_K - 2, 2):
            run_pipeline_stage(read_stage=0, next_k=k_iter + 1)
            run_pipeline_stage(read_stage=1, next_k=k_iter + 2)

        # Drain.
        run_pipeline_stage(read_stage=0, next_k=K // BLOCK_K - 1)
        run_pipeline_stage(read_stage=1, next_k=None, read_next=False)

        # Cast accumulator to bf16, store.
        mma_frag_C_bf16.store(
            fx.arith.trunc_f(fx.T.VectorType.get([C_PER_THREAD], fx.T.bf16()), mma_frag_C.load())
        )
        fx.copy(buffer_copy_16b, mma_frag_C_retile, thr_gC)

    @flyc.jit
    def nt_gemm_bf16(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Tiled g→r copy for A — 8 bf16 (= 16 B = b128) per thread per issue.
        val_per_thr = 8
        thrs_col    = BLOCK_K // val_per_thr           # 64 / 8 = 8
        thrs_row    = NUM_THREADS // thrs_col          # 512 / 8 = 64
        tiled_copy_g2s_A = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
            fx.make_layout(
                ((thrs_col, thrs_row), (1, val_per_thr)),
                ((thrs_row * val_per_thr, 1), (1, thrs_row)),
            ),
            fx.make_tile(thrs_row, BLOCK_K),
        )

        # 2 warps M, 4 warps N, 1 warp K — matches HIP v1.
        # Strides (NUM_WARP_N, 1, 0): consecutive physical warps differ in N
        # (stride 1), then jump NUM_WARP_N to get to the next M row.
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_N, MFMA_K, fx.BFloat16)),
            fx.make_layout((NUM_WARP_M, NUM_WARP_N, 1), (NUM_WARP_N, 1, 0)),
        )

        gemm_kernel(A, B, C, tiled_mma, tiled_copy_g2s_A).launch(
            grid=(M // BLOCK_M, N // BLOCK_N, 1),
            block=(NUM_THREADS, 1, 1),
            smem=SMEM_BYTES,
            stream=stream,
        )

    return nt_gemm_bf16


def run(M: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    """Entry point used by main.py."""
    fn = compile_nt_gemm(M, N, K)
    fn(A, B, C, stream=fx.Stream(torch.cuda.current_stream()))
