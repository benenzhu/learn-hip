














#include <hip/hip_runtime.h>
// #include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 32;

constexpr int LDA = K;
constexpr int LDB = K;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;

__device__ void print_mem(const _Float16 *ptr, int row=16, int col=16){
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

__global__ void fp16_gemm_16x16x16_NTN(const _Float16* A, const _Float16* B, _Float16* D)
{

  using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    // printf("Global A  : \n");
    // print_mem(A, 16, 16);
    // printf("Global B: \n");
    // print_mem(B, 16, 16);
  }
  __shared__ _Float16 shared_A[32 * 32]; // 32 * 32 / 64 = 16
  __shared__ _Float16 shared_B[32 * 32];
   
  float16x4 a;
  float16x4 b;
  auto gmem_to_smem = [&](const _Float16* gmem, _Float16* smem){
    constexpr int num_iters = (M * K) / (64/*thread*/ * 4/*index*/ * 2/*fp16->float*/);
    for(int i = 0; i < num_iters; i++){
      const int idx = (i * 64 + threadIdx.x) * 4 * 2;
      const float4 data = reinterpret_cast<const float4*>(gmem + idx)[0];
      reinterpret_cast<float4*>(smem +idx)[0] = data;
    }   
  };
  gmem_to_smem(A, shared_A);
  gmem_to_smem(B, shared_B);
  __syncthreads();
  for(int c_row = 0; c_row < 2; c_row++){
    for(int c_col = 0; c_col < 2; c_col++){
      floatx4 d = {0}; // zero out 4 vanilla VGPRs
      for(int i = 0; i < 2; i++){
        int a_row_index = c_row + i;         
        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);

      }  
      for(int i = 0; i < 4; i++){
        int lane = threadIdx.x % 16;
        int index = threadIdx.x / 16 * 4;
        const int d_idx = (c_row * 16 + index + i) * 32 + c_col * 16 + lane;
        D[d_idx] = d[i];
      }
    }
  }
  for(int i = 0; i < 4; ++i){
    const int a_idx =  threadIdx.x * LDA      // consecutive threads cover 16 consecutive rows
                     + i                      // consecutive registers take consecutive columns
                     + threadIdx.y * 4;       // groups of 16 lanes skip 4 columns
    a[i] = A[a_idx];

    const int b_idx =  threadIdx.x * LDA           // consecutive threads cover 16 consecutive columns
                     + i                // consecutive registers take consecutive rows
                     + threadIdx.y * 4; // groups of 16 lanes skip 4 rows
    b[i] = B[b_idx];
  }

  d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  //                                        ^  ^  ^
  //D(=C)                                   |  |  C(=D)
  //                      16 columns of A---|  |--- 16 rows of B

  for(int i = 0; i < 4; ++i){
    const int d_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                     + i * LDD                // consecutive registers take consecutive rows of 16 floats
                     + threadIdx.y * 4 * LDD; // groups of 16 lanes skip 4 rows

    D[d_idx] = d[i];
  }
  __syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    printf("Global D: \n");
    print_mem(D, 16, 16);
  }
}