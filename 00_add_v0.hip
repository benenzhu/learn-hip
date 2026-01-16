/*
编译命令：  hipcc -g add.hip
查看正常越界地址： 
rocgdb /a.out   # 然后执行 r 让他跑起来
会显示:
==============
==== Thread 5 "a.out" received signal SIGSEGV, Segmentation fault.
==== Warning: precise memory violation signal reporting is not enabled, reported
==== location may not be accurate.  See "show amdgpu precise-memory".
==== [Switching to thread 5, lane 0 (AMDGPU Lane 1:2:1:1/0 (0,0,0)[0,0,0])]
==== add_kernel (A=<optimized out>, B=<optimized out>, C=<optimized out>, N=1048576) at add.hip:25
==== 25          C[col] = A[col] + B[col];
==== (gdb) 

然后直接执行 ./a.out 会显示如下，但是无法定位错误....
    normal kernel done
    start illegal kernel...
    Memory access fault by GPU node-2 (Agent handle: 0x1c35430) on address 0x7ede51a00000. Reason: Unknown.
    GPU core dump created: gpucore.1835945
    fish: Job 1, './a.out' terminated by signal SIGABRT (Abort)

*/


#include <hip/hip_runtime.h>
__global__
__launch_bounds__(512)
void add_kernel(int *A, const int *B, int *C, int N) {
  // A, B, C are 2D matrices with shape M, N
  // we will use 1 threadblock to handle 1 row
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tb_size = blockDim.x;

  A += bid * N;
  B += bid * N;
  C += bid * N;
  for (int col = tid; col < N; col += tb_size)
    C[col] = A[col] + B[col];
}

#define HICHECK(call) { \
  hipError_t err = call; \
  if(err != hipSuccess) { \
    printf("Error: %s\n", hipGetErrorString(err)); \
    exit(1); \
  } \
}

int main(){
  int *d_A, *d_B, *d_C, *d_smallB;
  constexpr int N = 1024 * 1024;
  HICHECK(hipMalloc(&d_A, sizeof(int) * N));
  HICHECK(hipMalloc(&d_B, sizeof(int) * N));
  HICHECK(hipMalloc(&d_C, sizeof(int) * N));
  HICHECK(hipMalloc(&d_smallB, sizeof(int) * N / 1024));
  
  add_kernel<<<1, 512>>>(d_A, d_B, d_C, N);
  HICHECK(hipDeviceSynchronize());
  printf("normal kernel done\n");


  printf("start illegal kernel...\n");
  add_kernel<<<1, 512>>>(d_A, d_smallB, d_C, N);
  HICHECK(hipDeviceSynchronize());
  printf("illegal kernel done\n"); 
  return 0;
}