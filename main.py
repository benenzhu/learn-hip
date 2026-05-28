import os
# FlyDSL: dump every compile stage (incl. final .s) and skip the disk-cache so
# edits to the kernel source always retrigger codegen. Must be set BEFORE flydsl
# is imported anywhere.
os.environ.setdefault("FLYDSL_DUMP_IR", "1")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

from dataclasses import dataclass
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
  best = max(range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.mem_get_info(i)[0])      
  torch.cuda.set_device(best)
  print(f"[main] Selected GPU {best} (most free VRAM) used: {308556070912 - torch.cuda.mem_get_info(best)[0]}")
import time
import importlib
import rtc
import log as logmodule
log = logmodule.log
# os.environ["ROCPROF_COUNTER_COLLECTION"] = "1"

importlib.reload(rtc)
# import tritonblas
# from tritonblas.matmul import persistent_matmul_lt
# importlib.reload(tritonblas)
# from tritonblas.matmul import persistent_matmul_lt
# importlib.reload(tritonblas.matmul)
from rtc import _compile_kernel, get_triton_gemm_NTN, my_assert_close


torch.set_printoptions(threshold=1000, edgeitems=3, sci_mode=False)     
def get_kernel(kernel_name, file_name="00_add.hip", config=None):
    tic = time.time()
    source = open(file_name, "r").read()
    defines = []
    if config is not None:
        defines = ["#define PYTHON_CALL\n"]
        for key, value in vars(config).items():
            defines.append(f"#define {key} {value}\n")
        # Check if source starts with non-empty lines and warn
        source_lines = source.split('\n')
        non_empty_prefix = []
        for line in source_lines:
            if line.strip():
                non_empty_prefix.append(line)
            else:
                break
        
        if non_empty_prefix:
            log(f"Warning: Replacing {len(non_empty_prefix)} non-empty line(s) at the beginning of source file")
            log(source_lines[:len(non_empty_prefix)])
        
        # Replace the first few lines with defines
        num_lines_to_replace = len(defines)
        remaining_source = '\n'.join(source_lines[num_lines_to_replace:])
        source = "".join(defines) + remaining_source
    if len(defines) > 0:
        log("".join(defines))   

    kernel = _compile_kernel(
        kernel_source=source,
        kernel_name=kernel_name,
        nvcc_options=[
            "-std=c++20", 
            "-Dhip_rtc",
            "-g", 
            "-save-temps",
            "-ldl",
            "-lm",
            # "-fverbose-asm",
            "-Rpass-analysis=kernel-resource-usage"
            ],
        save_ptx=True,
    )
    toc = time.time()
    log(f"compile used {toc - tic}")
    return kernel





def test_add_kernel(): 
    M, N = 100, 2048
    A = torch.randint(-100, 100, (M, N * 100), device="cuda", dtype=torch.int32)
    B = torch.randint(-100, 100, (M, N * 100), device="cuda", dtype=torch.int32) 
    C = torch.empty_like(A)
    
    add_kernel = get_kernel("add_kernel", "00_add_v0.hip")
    print("start_kernel here", A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N)
    add_kernel((M, 1, 1), (256, 1, 1), (A, B, C, M, N))
    print("end_kernel here")
    torch.cuda.synchronize()
    print("synchronize done")
    # torch.testing.assert_close(C, A + B)
    print("test passed")
    os.system("python gen_pure.py add_kernel-hip-amdgcn-amd-amdhsa-gfx942:sramecc+:xnack-.s")
    return C

# test_add_kernel()


def test_add_kernel_v2(): 
    M, N = 100, 2048
    A = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
    B = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32) 
    C = torch.empty_like(A)
    
    add_kernel = get_kernel("add_kernel", "00_add_v1_builtin_assume.hip")
    print("start_kernel here", A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N)
    add_kernel((M, 1, 1), (256, 1, 1), (A, B, C, M, N))
    print("end_kernel here")
    torch.cuda.synchronize()
    print("synchronize done")
    torch.testing.assert_close(C, A + B)
    print("test passed")
    return C
    
# test_add_kernel_v2()

def test_bf16_matmul_NNN():
    matmul_kernel = get_kernel("fp16_gemm_16x16x16_NNN", "01_mfma.hip")  
    print(matmul_kernel)
    A = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    B = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    C = torch.zeros(16*16, device="cuda").reshape(16, 16).half()
    matmul_kernel((1,1,1), (16,4,1), (A, B, C))
    torch.cuda.synchronize()

# test_bf16_matmul_NNN()


def test_bf16_matmul_NTN():
    matmul_kernel = get_kernel("fp16_gemm_16x16x16_NTN", "01_mfma.hip")  
    # print(matmul_kernel)
    A = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    B = torch.arange(16*16, device="cuda").reshape(16, 16).half().transpose(0, 1).contiguous()*0.1
    C = torch.zeros(16*16, device="cuda").reshape(16, 16).half()
    matmul_kernel((1,1,1), (16,4,1), (A, B, C))
    torch.cuda.synchronize()
    
    
def test_bf16_matmul_NTN():
    matmul_kernel = get_kernel("fp16_gemm_16x16x16_NTN", "01_mfma.hip")  
    # print(matmul_kernel)
    A = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    B = torch.arange(16*16, device="cuda").reshape(16, 16).half().transpose(0, 1).contiguous()*0.1
    C = torch.zeros(16*16, device="cuda").reshape(16, 16).half()
    matmul_kernel((1,1,1), (16,4,1), (A, B, C))
    torch.cuda.synchronize()


# test_bf16_matmul_NTN()


def bench(f, A, B, C, check_correct=True):
    import inspect
    frame = inspect.currentframe().f_back
    name = frame.f_code.co_name
    from triton.testing import do_bench
    f()

    M, N, K = A.shape[0], B.shape[0], A.shape[1]
    torch.cuda.synchronize()
    right_output = torch.zeros_like(C)
    if M >= 51200:
        A = A.T.contiguous()
        BT = B.T.contiguous()
        triton_fn = lambda: persistent_matmul_lt(A.T, BT, right_output, None)
    else:
        BT = B.T
        triton_fn = lambda: get_triton_gemm_NTN(A, B, right_output, M, N, K)
        # triton_fn = lambda: torch.matmul(A, B.T, out = right_output)
    if check_correct:
        ret = triton_fn()
        if my_assert_close(C, right_output) is not None:
            log(f"{name} failed")
            return C, right_output
    if "ROCPROF_COUNTER_COLLECTION" in os.environ:
        print("ROC_PERF on fast return here.")
        return ret
    torch.cuda.synchronize()
    latency_ms = do_bench(f, warmup=100, rep=500, return_mode="median")
    tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
    log(f"{name}: {tflops:.2f} TFLOPS")
    latency_ms = do_bench(triton_fn, warmup=100, rep=500, return_mode="median")
    tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
    log(f"triton: \t{tflops:.2f} TFLOPS")
    return C, right_output

@dataclass
class Bf16MatmulFullNTNConfig:
    M: int = 4096
    N: int = 4096
    K: int = 4096
    NUM_WARP_M: int = 2
    NUM_WARP_N: int = 2
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    BLOCK_K: int = 64
    SMEM_STRIDE: int = 0
    def get_grid_size(self):
        return ((self.M + self.BLOCK_M - 1) // self.BLOCK_M)* ((self.N + self.BLOCK_N - 1) // self.BLOCK_N)
    def get_tb_size(self):
        return 64 * self.NUM_WARP_M * self.NUM_WARP_N
    def get_shared_mem(self):
        if not self.SMEM_STRIDE:
            self.SMEM_STRIDE = self.BLOCK_K
        return (self.BLOCK_M + self.BLOCK_N) * self.SMEM_STRIDE * 2


def get_inputNTN(M, N, K):
    if M <= 512:
        A = torch.arange(M*K, device="cuda").reshape(M, K).bfloat16().contiguous() * 0.01
        B = torch.arange(N*K, device="cuda").reshape(N, K).bfloat16().contiguous() * 0.01
    else:
        A = torch.randn(M, K, device="cuda").bfloat16().contiguous() * 0.1
        B = torch.randn(N, K, device="cuda").bfloat16().contiguous() * 0.1
    C = torch.zeros(M, N, device="cuda").bfloat16().contiguous()
    return A, B, C


def bf16_matmul_full_NTN(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(M=M, N=N, K=K)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN", "02_fp16_gemm_v1_NTN.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    bench(kernel_fn, A, B, C)
    
# ret = bf16_matmul_full_NTN(256, 256, 256)
# ret = bf16_matmul_full_NTN(4864, 4096, 4096)


def bf16_matmul_full_NTN_v2(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(M=M, N=N, K=K, SMEM_STRIDE=64 + 8)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN", "02_fp16_gemm_v1.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    bench(kernel_fn, A, B, C)
    
# ret = bf16_matmul_full_NTN_v2(4864, 4096, 4096)


def bf16_matmul_full_NTN_v3(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(M=M, N=N, K=K, SMEM_STRIDE=64)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN_v2", "02_fp16_gemm_v2.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    bench(kernel_fn, A, B, C)
    
# ret = bf16_matmul_full_NTN_v3(4864, 4096, 4096)

# print(list(os.environ.keys()))

def bf16_matmul_full_NTN_v2_opt1(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    

    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN_v3", "02_fp16_gemm_v3_NTN.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    
# ret = bf16_matmul_full_NTN_v2_opt1(4864, 4096, 4096)
# ret = bf16_matmul_full_NTN_v2_opt1(256, 256, 64)


def bf16_matmul_full_NTN_v4(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    

    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN_v4", "02_fp16_gemm_full_NTN_v4.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    
# ret = bf16_matmul_full_NTN_v2_opt1(4864, 4096, 4096)
# ret = bf16_matmul_full_NTN_v2_opt1(256, 256, 128)
# ret = bf16_matmul_full_NTN_v4(256, 256, 64)
# ret = bf16_matmul_full_NTN_v4(64*4, 64*4, 128*4)


def _03_fp16_gemm_v0(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("_3_fp16_gemm_v0", "03_fp16_gemm_v0.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    log(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    # matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K))
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    

# _03_fp16_gemm_v0: 138.06 TFLOPS
# ret = _03_fp16_gemm_v0(4864, 4096, 4096) 




def _03_fp16_gemm_v2(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("_3_fp16_gemm_v0", "03_fp16_gemm_v2.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    log(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    # matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K))
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    

# _03_fp16_gemm_v0: 138.06 TFLOPS
# ret = _03_fp16_gemm_v2(4864, 4096, 4096) 

def _03_fp16_gemm_v4(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("_3_fp16_gemm_v4", "03_fp16_gemm_v4.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    log(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    # matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K))
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    

# _03_fp16_gemm_v0: 138.06 TFLOPS
# ret = _03_fp16_gemm_v4(4864, 4096, 4096) 

def _03_fp16_gemm_v5(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("_3_fp16_gemm_v5", "03_fp16_gemm_v5.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    log(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    # matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K))
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    

# ret = _03_fp16_gemm_v5(4864, 4096, 4096) 

def cal_ratio(diff):
    log("diff", diff)
    log("diff ratio", (diff.abs() > 0.0001).sum().item() / diff.numel() * 100, "%")


def _03_fp16_gemm_v6(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("_3_fp16_gemm_v6", "03_fp16_gemm_v6.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    log(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    # matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K))
    # profiler = torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CUDA
    #     ],
    #     record_shapes=False,
    #     with_stack=False, 
    # ) 
    # with profiler:
    ret = bench(kernel_fn, A, B, C)
    os.system("python gen_pure.py _3_fp16_gemm_v6-hip-amdgcn-amd-amdhsa-gfx942:sramecc+:xnack-.s")
    # profiler.export_chrome_trace("03_fp16_gemm_v6_trace.json")
    return ret
    

# ret = _03_fp16_gemm_v6(256,256,64) 
# ret = _03_fp16_gemm_v6(4864, 4096, 4096) 
# ret = _03_fp16_gemm_v6(4864, 4096, 8192) 



def _03_fp16_gemm_v7(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("_3_fp16_gemm_v7", "03_fp16_gemm_v7.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    log(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K))
    ret = bench(kernel_fn, A, B, C)
    os.system("python gen_pure.py _3_fp16_gemm_v7-hip-amdgcn-amd-amdhsa-gfx942:sramecc+:xnack-.s")
    return ret
    

# ret = _03_fp16_gemm_v7(4864, 4096, 4096)  # for mi300x
# ret = _03_fp16_gemm_v7(4096, 4096, 4096)  # for mi355x


def _03_fp16_gemm_v8(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("_3_fp16_gemm_gfx950_v1", "03_fp16_gemm_gfx950_v1.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    log(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K))
    ret = bench(kernel_fn, A, B, C)
    os.system("python gen_pure.py _3_fp16_gemm_v7-hip-amdgcn-amd-amdhsa-gfx950:sramecc+:xnack-.s")
    return ret
    

# ret = _03_fp16_gemm_v7(4864, 4096, 4096)  # for mi300x
# ret = _03_fp16_gemm_v8(4096, 4096, 4096)  # for mi355x


def _04_nt_gemm_flydsl(M, N, K):
    """FlyDSL port of 03_fp16_gemm_gfx950_v1 — module name starts with a digit
    so we load via importlib instead of `import 04_nt_gemm_flyDsl_gfx950`."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nt_gemm_flydsl_gfx950", "04_nt_gemm_flyDsl_gfx950.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    A, B, C = get_inputNTN(M, N, K)
    # Warm-up + JIT compile (cached by lru_cache on (M,N,K)).
    mod.run(M, N, K, A, B, C)
    log(f"GRID=({M//mod.BLOCK_M},{N//mod.BLOCK_N},1), TB={mod.NUM_THREADS}, smem={mod.SMEM_BYTES}")
    kernel_fn = lambda: mod.run(M, N, K, A, B, C)
    ret = bench(kernel_fn, A, B, C)
    return ret


# ret = _04_nt_gemm_flydsl(4096, 4096, 4096)  # FlyDSL NT GEMM, gfx950 (v1, layout style)


def _04_nt_gemm_flydsl_v2(M, N, K):
    """FlyDSL HIP-style imperative port of 03_fp16_gemm_gfx950_v1.hip — K=32 native.
    Set FLYDSL_DUMP_IR=1 to also produce 04_nt_gemm_flyDsl_gfx950_v2.spure.s
    alongside the HIP .spure.s files."""
    import importlib.util, glob, shutil
    spec = importlib.util.spec_from_file_location(
        "nt_gemm_flydsl_gfx950_v2", "04_nt_gemm_flyDsl_gfx950_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    A, B, C = get_inputNTN(M, N, K)
    mod.run(M, N, K, A, B, C)
    log(f"GRID=({M//mod.BLOCK_M},{N//mod.BLOCK_N},1), TB={mod.BLOCK_THREADS}")
    kernel_fn = lambda: mod.run(M, N, K, A, B, C)
    ret = bench(kernel_fn, A, B, C)

    # Pipe the FlyDSL final ISA through gen_pure.py if FLYDSL_DUMP_IR was set.
    # FlyDSL writes per-kernel subdirs under FLYDSL_DUMP_DIR (default ~/.flydsl/debug).
    dump_root = os.environ.get(
        "FLYDSL_DUMP_DIR", os.path.expanduser("~/.flydsl/debug")
    )
    isa_glob = os.path.join(dump_root, "nt_gemm_bf16_*", "17_final_isa.s")
    matches = sorted(glob.glob(isa_glob), key=os.path.getmtime, reverse=True)
    if matches:
        src = matches[0]
        dst = "04_nt_gemm_flyDsl_gfx950_v2.s"
        shutil.copyfile(src, dst)
        os.system(f"python gen_pure.py {dst}")
        log(f"FlyDSL ASM: {dst}  (cleaned: {dst}pure.s)")
    return ret


# ret = _04_nt_gemm_flydsl_v2(8192, 8192, 8192)  # FlyDSL NT GEMM v2 — match HK 8192³ benchmark

def fp4_gemm_atom_323264(): 
    atom_kernel = get_kernel(
        "mfma_fp32_32x32x64_fp4_fp4",
        "05_fp4_gemm_atom.hip",
    )

    torch.manual_seed(5)
    # Logical operands for one MFMA atom:
    #   A: [32, 64] fp4 e2m1
    #   B: [64, 32] fp4 e2m1
    # Each uint8 stores two fp4 values: low nibble then high nibble.
    a_nibbles = torch.randint(0, 16, (32, 64), dtype=torch.uint8)
    b_nibbles = torch.randint(0, 16, (64, 32), dtype=torch.uint8)

    A = (a_nibbles[:, 0::2] | (a_nibbles[:, 1::2] << 4)).contiguous().cuda()

    # Match the B layout consumed by 05_fp4_gemm_atom.hip:
    #   byte offset = k_half * 512 + k_inner * 16 + n_pair
    #   low/high nibble hold adjacent N columns.
    B_host = torch.empty((64 * 32 // 2,), dtype=torch.uint8)
    for k in range(64):
        k_half = k // 32
        k_inner = k % 32
        for n_pair in range(16):
            B_host[k_half * 512 + k_inner * 16 + n_pair] = (
                b_nibbles[k, 2 * n_pair] | (b_nibbles[k, 2 * n_pair + 1] << 4)
            )
    B = B_host.cuda()
    C = torch.empty((32, 32), device="cuda", dtype=torch.float32)

    atom_kernel((1, 1, 1), (64, 1, 1), (A, B, C))
    torch.cuda.synchronize()

    fp4_to_f32 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
        dtype=torch.float32,
    )
    expected = (
        fp4_to_f32[a_nibbles.cuda().long()]
        @ fp4_to_f32[b_nibbles.cuda().long()]
    )
    torch.testing.assert_close(C, expected, rtol=0, atol=0)
    log("fp4_gemm_atom passed: random fp4 layout matches torch reference")
    return C

# ret = fp4_gemm_atom_323264()

def fp4_gemm_atom_1616128():
    atom_kernel = get_kernel(
        "mfma_fp32_16x16x128_fp4_fp4",
        "05_fp4_gemm_atom.hip",
    )

    torch.manual_seed(6)
    # Logical operands for one MFMA atom:
    #   A: [16, 128] fp4 e2m1
    #   B: [128, 16] fp4 e2m1
    a_nibbles = torch.randint(0, 16, (16, 128), dtype=torch.uint8)
    b_nibbles = torch.randint(0, 16, (128, 16), dtype=torch.uint8)
    a_scales = torch.randint(124, 128, (16, 4), dtype=torch.uint8)
    b_scales = torch.randint(124, 128, (16, 4), dtype=torch.uint8)

    A = (a_nibbles[:, 0::2] | (a_nibbles[:, 1::2] << 4)).contiguous().cuda()

    # Match mfma_fp32_16x16x128_fp4_fp4:
    #   byte offset = k_lane * 32 * 8 + k_inner * 8 + n_pair
    #   low/high nibble hold adjacent N columns.
    B_host = torch.empty((128 * 16 // 2,), dtype=torch.uint8)
    for k in range(128):
        k_lane = k // 32
        k_inner = k % 32
        for n_pair in range(8):
            B_host[k_lane * 32 * 8 + k_inner * 8 + n_pair] = (
                b_nibbles[k, 2 * n_pair] | (b_nibbles[k, 2 * n_pair + 1] << 4)
            )
    B = B_host.cuda()
    # Scale operands are lane-local for this atom:
    #   offset = k_lane * 16 + lane_mn
    # where k_lane selects one 32-wide K quarter.
    AScale = a_scales.T.contiguous().cuda()
    BScale = b_scales.T.contiguous().cuda()
    C = torch.empty((16, 16), device="cuda", dtype=torch.float32)

    atom_kernel((1, 1, 1), (64, 1, 1), (A, B, AScale, BScale, C))
    torch.cuda.synchronize()

    fp4_to_f32 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
        dtype=torch.float32,
    )
    a_f32 = fp4_to_f32[a_nibbles.cuda().long()]
    b_f32 = fp4_to_f32[b_nibbles.cuda().long()]
    a_scale_f32 = 2 ** (a_scales.cuda().repeat_interleave(32, dim=1).to(torch.float32) - 127)
    b_scale_f32 = 2 ** (
        b_scales.cuda().T.repeat_interleave(32, dim=0).to(torch.float32) - 127
    )
    expected = (a_f32 * a_scale_f32) @ (b_f32 * b_scale_f32)
    torch.testing.assert_close(C, expected, rtol=0, atol=1e-5)
    log("fp4_gemm_atom_1616128 passed: random fp4 data and E8M0 scale layout match torch reference")
    return C


# fp4_gemm_atom_1616128()


def fp4_gemm_tile_3232256_packed_scale(
    kernel_name="mfma_fp32_32x32x256_fp4_fp4_packed_scale",
    log_label="fp4_gemm_tile_3232256_packed_scale",
):
    tile_kernel = get_kernel(
        kernel_name,
        "05_fp4_gemm_atom.hip",
    )

    torch.manual_seed(7)
    # Logical tile assembled from 8 x 16x16x128 atoms:
    #   2 M halves x 2 N halves x 2 K halves.
    a_nibbles = torch.randint(0, 16, (32, 256), dtype=torch.uint8)
    b_nibbles = torch.randint(0, 16, (256, 32), dtype=torch.uint8)
    a_scales = torch.randint(124, 128, (32, 8), dtype=torch.uint8)
    b_scales = torch.randint(124, 128, (32, 8), dtype=torch.uint8)

    A = (a_nibbles[:, 0::2] | (a_nibbles[:, 1::2] << 4)).contiguous().cuda()

    # B is logical [K, N], packed along N pairs.
    B_host = torch.empty((256, 16), dtype=torch.uint8)
    for k in range(256):
        for n_pair in range(16):
            B_host[k, n_pair] = (
                b_nibbles[k, 2 * n_pair] | (b_nibbles[k, 2 * n_pair + 1] << 4)
            )
    B = B_host.contiguous().cuda()

    def pack_scale_32x8(scales: torch.Tensor):
        words = torch.empty((4, 16), dtype=torch.int32)
        for kg_inner in range(4):
            for inner in range(16):
                words[kg_inner, inner] = (
                    scales[inner, kg_inner].to(torch.int32)
                    | (scales[16 + inner, kg_inner].to(torch.int32) << 8)
                    | (scales[inner, 4 + kg_inner].to(torch.int32) << 16)
                    | (scales[16 + inner, 4 + kg_inner].to(torch.int32) << 24)
                )
        return words.contiguous()

    AScale = pack_scale_32x8(a_scales).cuda()
    BScale = pack_scale_32x8(b_scales).cuda()
    C = torch.empty((32, 32), device="cuda", dtype=torch.float32)

    tile_kernel((1, 1, 1), (64, 1, 1), (A, B, AScale, BScale, C))
    torch.cuda.synchronize()

    fp4_to_f32 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
        dtype=torch.float32,
    )
    a_f32 = fp4_to_f32[a_nibbles.cuda().long()]
    b_f32 = fp4_to_f32[b_nibbles.cuda().long()]
    a_scale_f32 = 2 ** (a_scales.cuda().repeat_interleave(32, dim=1).to(torch.float32) - 127)
    b_scale_f32 = 2 ** (
        b_scales.cuda().T.repeat_interleave(32, dim=0).to(torch.float32) - 127
    )
    expected = (a_f32 * a_scale_f32) @ (b_f32 * b_scale_f32)
    torch.testing.assert_close(C, expected, rtol=0, atol=1e-5)
    log(f"{log_label} passed: packed scale word uses all four bytes")
    return C


def fp4_gemm_tile_3232256_scale_preshuffle():
    return fp4_gemm_tile_3232256_packed_scale(
        kernel_name="mfma_fp32_32x32x256_fp4_fp4_scale_preshuffle",
        log_label="fp4_gemm_tile_3232256_scale_preshuffle",
    )