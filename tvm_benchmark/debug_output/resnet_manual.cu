__device__ inline long max(int a, long b)
{
  return max((long)a, b);
}
__device__ inline long max(long a, int b)
{
  return max(b, a);
}
__device__ inline long min(long a, int b)
{
  return min(a, (long)b);
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>
#endif
#include <mma.h>
__device__ void nvcuda::wmma::mma_sync(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int, void> &d,
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> &a,
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> &b,
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int, void> &c) 
{
  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);
  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);
  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));  
}

extern "C" __global__ void fused_nn_global_avg_pool2d_cast_cast_left_shift_multiply_add_right_shift_cast_cl_10830175937736834516__kernel0(void* __restrict__ placeholder, void* __restrict__ T_cast) {
  int tensor[1];
  tensor[(0)] = 0;
  for (int rv0 = 0; rv0 < 7; ++rv0) {
    for (int rv1 = 0; rv1 < 7; ++rv1) {
      tensor[(0)] = (tensor[(0)] + ((int*)placeholder)[((((((((int)threadIdx.y) * 100352) + (rv0 * 14336)) + (rv1 * 2048)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))]);
    }
  }
  ((signed char*)T_cast)[((((((int)threadIdx.y) * 2048) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)(tensor[(0)] / 49)) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 127), -128));
}

extern "C" __global__ void fused_nn_pad_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    if ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 331776) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 41472) {
        ((int*)T_pad)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 36864) * 4608) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 9) * 512)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) / 8)))] = (((((36864 <= (((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner)) && ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 294912)) && (1 <= (((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 9))) && ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 9) < 8)) ? ((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 36864) * 3584) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 9) * 512)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) / 8)) - 4096))] : 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__kernel1(unsigned int* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_cast, void* __restrict__ placeholder1) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[384];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[32];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int kh = 0; kh < 3; ++kh) {
    for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
      for (int ax1 = 0; ax1 < 3; ++ax1) {
        for (int ax3 = 0; ax3 < 4; ++ax3) {
          if (((int)threadIdx.z) < 1) {
            ((int*)compute_shared)[(((((ax1 * 128) + (((int)threadIdx.z) * 128)) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)compute)[((((((((((((int)blockIdx.z) / 7) * 4608) + (kh * 4608)) + (ax1 * 512)) + (((int)threadIdx.z) * 512)) + ((((int)blockIdx.z) % 7) * 512)) + (ic_outer * 128)) + (ax3 * 32)) + ((int)threadIdx.x)))];
          }
        }
      }
      __syncthreads();
      for (int kw = 0; kw < 3; ++kw) {
        for (int ax31 = 0; ax31 < 4; ++ax31) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + ((kw * 1024) + (ax31 * 256)) / 8), 32);
        }
        __syncthreads();
        for (int ax2 = 0; ax2 < 16; ++ax2) {
          for (int ax3_inner_inner = 0; ax3_inner_inner < 2; ++ax3_inner_inner) {
            for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
              ((int*)placeholder_shared)[(((((ax2 * 128) + (((int)threadIdx.z) * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((kh * 98304) + (kw * 32768)) + (((int)blockIdx.y) * 8192)) + (ax2 * 512)) + (ic_outer * 128)) + (((int)threadIdx.z) * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
            }
          }
        }
        __syncthreads();
        for (int ax21 = 0; ax21 < 8; ++ax21) {
          for (int ax32 = 0; ax32 < 4; ++ax32) {
            (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 8192) + (ax21 * 1024)) + (ax32 * 256)) / 8), 32);
          }
        }
        for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
          for (int o_c = 0; o_c < 8; ++o_c) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
  }
  __syncthreads();
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 512) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((unsigned int*)T_cast)[(((((((((int)blockIdx.z) * 512) + (ax2_inner_ax3_inner_fused_outer * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.z) * 8)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 64)) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[16];
  __shared__ unsigned int compute_shared[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  __shared__ int Conv[1024];
  #pragma unroll
  for (int o_c_init = 0; o_c_init < 16; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 16; ++ic_outer) {
    #pragma unroll
    for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
      #pragma unroll
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
        ((int*)compute_shared)[(((int)threadIdx.x))] = ((int*)placeholder)[((((((((((int)blockIdx.z) / 7) * 28672) + ((((int)blockIdx.z) % 7) * 2048)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 128)) + (ic_outer * 8)) + (ic_inner * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
      }
      __syncthreads();
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + 0 / 8), 32);
      #pragma unroll
      for (int ax2 = 0; ax2 < 16; ++ax2) {
        #pragma unroll
        for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
          ((int*)placeholder_shared)[(((ax2 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 16384) + (ax2 * 1024)) + (ic_outer * 64)) + (ic_inner * 32)) + ((int)threadIdx.x)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 16; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
      }
      #pragma unroll
      for (int o_c = 0; o_c < 16; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  #pragma unroll
  for (int o_inner = 0; o_inner < 16; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int ax3_outer_inner = 0; ax3_outer_inner < 16; ++ax3_outer_inner) {
    #pragma unroll
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((int*)T_cast)[(((((((((int)blockIdx.z) * 16384) + (ax2_inner_ax3_inner_fused_outer * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = (Conv[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_multiply_add_right_shift_cast_nn_relu_kernel1(void* __restrict__ placeholder, signed char* __restrict__ kernel_im2col_pack, int* __restrict__ gemm_C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> gemm_C_wmma_accumulator[2];
  __shared__ signed char data_im2col_pack_shared[2048];
  __shared__ signed char kernel_im2col_pack_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> data_im2col_pack_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> kernel_im2col_pack_shared_wmma_matrix_b[2];
  for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
    (void)nvcuda::wmma::fill_fragment(gemm_C_wmma_accumulator[j_c_init], 0.000000e+00f);
  }
  for (int k1_outer = 0; k1_outer < 5; ++k1_outer) {
    __syncthreads();
    for (int ax2_ax3_fused_inner_inner_s = 0; ax2_ax3_fused_inner_inner_s < 8; ++ax2_ax3_fused_inner_inner_s) {
      data_im2col_pack_shared[(((((((int)threadIdx.y) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)) + ax2_ax3_fused_inner_inner_s))] = (((((k1_outer * 32) + (((int)threadIdx.z) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) & 15)) < 147) ? ((signed char*)placeholder)[((((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) >> 4)) / 12544) * 158700) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) >> 4)) % 12544) / 112) * 1380)) + (((((k1_outer * 32) + (((int)threadIdx.z) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) & 15)) / 21) * 690)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) >> 4)) % 112) * 6)) + ((((k1_outer * 32) + (((int)threadIdx.z) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) & 15)) % 21)))] : (signed char)0);
    }
    ((int2*)(kernel_im2col_pack_shared + ((((((int)threadIdx.y) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)))))[0] = ((int2*)(kernel_im2col_pack + (((((((int)threadIdx.y) * 2560) + (k1_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)))))[0];
    __syncthreads();
    for (int k1_inner = 0; k1_inner < 2; ++k1_inner) {
      (void)nvcuda::wmma::load_matrix_sync(data_im2col_pack_shared_wmma_matrix_a[0], ((signed char *)data_im2col_pack_shared + ((((int)threadIdx.y) * 512) + (k1_inner * 256))), 16);
      for (int ax0 = 0; ax0 < 2; ++ax0) {
        (void)nvcuda::wmma::load_matrix_sync(kernel_im2col_pack_shared_wmma_matrix_b[ax0], ((signed char *)kernel_im2col_pack_shared + (((((int)threadIdx.z) * 1024) + (ax0 * 512)) + (k1_inner * 256))), 16);
      }
      for (int j_c = 0; j_c < 2; ++j_c) {
        (void)nvcuda::wmma::mma_sync(gemm_C_wmma_accumulator[j_c], data_im2col_pack_shared_wmma_matrix_a[0], kernel_im2col_pack_shared_wmma_matrix_b[j_c], gemm_C_wmma_accumulator[j_c]);
      }
    }
  }
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)gemm_C + ((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (j_inner * 256))), gemm_C_wmma_accumulator[j_inner], 16, nvcuda::wmma::mem_row_major);
  }
}

extern "C" __global__ void fused_transpose_kernel0(void* __restrict__ T_transpose, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 128) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) >> 11)) < 392) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 802816) {
        ((int*)T_transpose)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 128) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) >> 11)) % 49) * 16384) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 128) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) >> 11)) / 49) * 2048)) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) & 2047)))];
      }
    }
  }
}

extern "C" __global__ void fused_transpose_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    ((unsigned int*)T_cast)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) / 8)))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)) + 1))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)) + 2))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)) + 3))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)) + 4))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)) + 5))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)) + 6))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)((int*)placeholder)[((((((((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) >> 6) * 200704) + (((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 28672) * 3584)) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 56) * 64)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 63) >> 3) * 8)) + 7))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_multiply_add_right_shift_cast_nn_relu_kernel0(signed char* __restrict__ kernel_im2col_pack, void* __restrict__ placeholder) {
  kernel_im2col_pack[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = (((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 10) * 16) + (((int)threadIdx.x) & 15)) < 147) ? ((signed char*)placeholder)[(((((((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 10) * 16) + (((int)threadIdx.x) & 15)) / 3) * 192) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 10) * 48)) + (((((int)threadIdx.x) & 255) >> 4) * 3)) + ((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 10) * 16) + (((int)threadIdx.x) & 15)) % 3)))] : (signed char)0);
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[32];
  #pragma unroll
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        #pragma unroll
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
          ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((int)blockIdx.z) * 1024) + (ax0_inner_ax1_fused_inner * 256)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 32)) + (ic_outer * 16)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax31 = 0; ax31 < 4; ++ax31) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        #pragma unroll
        for (int ax3_inner_inner = 0; ax3_inner_inner < 4; ++ax3_inner_inner) {
          ((int*)placeholder_shared)[((((ax2 * 128) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 2048) + (ax2 * 256)) + (ic_outer * 128)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 4; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + ((ax21 * 1024) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 8; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((int*)T_relu)[((((((((((int)blockIdx.z) * 32768) + (ax0_inner_ax1_fused_inner * 8192)) + (ax2_inner_ax3_inner_fused_outer * 4096)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((((int*)placeholder_shared)[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int*)placeholder3)[((((((((((int)blockIdx.z) * 32768) + (ax0_inner_ax1_fused_inner * 8192)) + (ax2_inner_ax3_inner_fused_outer * 4096)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]), 2147483647), 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__5_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[16];
  __shared__ unsigned int compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[2];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[32];
  for (int o_c_init = 0; o_c_init < 16; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ax3 = 0; ax3 < 16; ++ax3) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
      if (((int)threadIdx.z) < 1) {
        ((int*)compute_shared)[((((((int)threadIdx.z) * 512) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((((int)blockIdx.z) / 14) * 28672) + ((((int)blockIdx.z) % 14) * 1024)) + (((int)threadIdx.z) * 512)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 64)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
      }
    }
  }
  __syncthreads();
  for (int ic_outer = 0; ic_outer < 8; ++ic_outer) {
    for (int ax31 = 0; ax31 < 2; ++ax31) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + ((ic_outer * 512) + (ax31 * 256)) / 8), 32);
    }
    __syncthreads();
    for (int ax2 = 0; ax2 < 32; ++ax2) {
      ((int*)placeholder_shared)[((((ax2 * 64) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((ax2 * 512) + (ic_outer * 64)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 16; ++ax21) {
      for (int ax32 = 0; ax32 < 2; ++ax32) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 2) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 8192) + (ax21 * 512)) + (ax32 * 256)) / 8), 32);
      }
    }
    for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
      for (int o_c = 0; o_c < 16; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 2) + ic_inner)], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  __syncthreads();
  for (int o_inner = 0; o_inner < 16; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 1024) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 16; ++ax3_outer_inner) {
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((unsigned int*)T_cast)[((((((((int)blockIdx.z) * 256) + (ax2_inner_ax3_inner_fused_outer * 128)) + ((((int)threadIdx.x) >> 3) * 32)) + (((int)threadIdx.z) * 16)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[(((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 1024) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 128) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[32];
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3 = 0; ax3 < 16; ++ax3) {
      #pragma unroll
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
        ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) / 14) * 28672) + ((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) % 14) * 1024)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 64)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
      }
    }
    __syncthreads();
    for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
      #pragma unroll
      for (int ax31 = 0; ax31 < 4; ++ax31) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + ((ic_outer * 1024) + (ax31 * 256)) / 8), 32);
      }
      __syncthreads();
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        #pragma unroll
        for (int ax3_inner_inner = 0; ax3_inner_inner < 4; ++ax3_inner_inner) {
          #pragma unroll
          for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
            ((int*)placeholder_shared)[((((ax2 * 128) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 4096) + (ax2 * 512)) + (ic_outer * 128)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 4; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + ((ax21 * 1024) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 8; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    #pragma unroll
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)compute_shared + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((int*)T_cast)[((((((((((int)blockIdx.z) * 32768) + (ax0_inner_ax1_fused_inner * 8192)) + (ax2_inner_ax3_inner_fused_outer * 4096)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = (((int*)compute_shared)[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__10_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[8];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[32];
  #pragma unroll
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 2; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ax3 = 0; ax3 < 8; ++ax3) {
      #pragma unroll
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
        if (((int)threadIdx.z) < 1) {
          ((int*)compute_shared)[((((((int)threadIdx.z) * 256) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((int)blockIdx.z) * 512) + (ax0_inner_ax1_fused_inner * 256)) + (((int)threadIdx.z) * 256)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 32)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ax31 = 0; ax31 < 8; ++ax31) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
    }
    #pragma unroll
    for (int ax2 = 0; ax2 < 8; ++ax2) {
      #pragma unroll
      for (int ax3_inner_inner = 0; ax3_inner_inner < 4; ++ax3_inner_inner) {
        ((int*)placeholder_shared)[(((((ax2 * 256) + (((int)threadIdx.z) * 128)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((ax2 * 256) + (((int)threadIdx.z) * 128)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ax21 = 0; ax21 < 4; ++ax21) {
      #pragma unroll
      for (int ax32 = 0; ax32 < 8; ++ax32) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 8) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 8192) + (ax21 * 2048)) + (ax32 * 256)) / 8), 32);
      }
    }
    #pragma unroll
    for (int ic_inner = 0; ic_inner < 8; ++ic_inner) {
      #pragma unroll
      for (int o_c = 0; o_c < 4; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 8) + ic_inner)], Conv_wmma_accumulator[o_c]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 4; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 256) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((unsigned int*)T_cast)[(((((((((int)blockIdx.z) * 128) + (ax0_inner_ax1_fused_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + (((int)threadIdx.z) * 4)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[(((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 32) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_max_pool2d_kernel0(void* __restrict__ placeholder, void* __restrict__ tensor) {
  int tensor_local[1];
  tensor_local[(0)] = -2147483648;
  for (int rv = 0; rv < 3; ++rv) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor_local[(0)] = max(tensor_local[(0)], (((1 <= ((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 3136) / 56) * 2) + rv)) && (1 <= (((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 56) * 2) + rv1))) ? ((int*)placeholder)[((((((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 56) * 14336) + (rv * 7168)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 56) * 128)) + (rv1 * 64)) + (((int)threadIdx.x) & 63)) - 7232))] : -2147483648));
    }
  }
  ((int*)tensor)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = tensor_local[(0)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__7_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[8];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[64];
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 8; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 8; ++ax3) {
        #pragma unroll
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
          ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((int)blockIdx.z) * 4096) + (ax0_inner_ax1_fused_inner * 512)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 64)) + (ic_outer * 32)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax31 = 0; ax31 < 8; ++ax31) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        #pragma unroll
        for (int ax3_inner_inner = 0; ax3_inner_inner < 8; ++ax3_inner_inner) {
          ((int*)placeholder_shared)[((((ax2 * 256) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 4096) + (ax2 * 512)) + (ic_outer * 256)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 8; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 8) + ax32)], ((int *)placeholder_shared + ((ax21 * 2048) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 8; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 8; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 8) + ic_inner)], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((unsigned int*)T_cast)[(((((((((int)blockIdx.z) * 1024) + (ax0_inner_ax1_fused_inner * 128)) + (ax2_inner_ax3_inner_fused_outer * 64)) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)blockIdx.y) * 8)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[(((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__4_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  #pragma unroll
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 2; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3 = 0; ax3 < 32; ++ax3) {
      #pragma unroll
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
        if (((int)threadIdx.z) < 1) {
          ((int*)compute_shared)[((((((int)threadIdx.z) * 1024) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((int)blockIdx.z) * 2048) + (ax0_inner_ax1_fused_inner * 1024)) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 128)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 8; ++ic_outer) {
      #pragma unroll
      for (int ax31 = 0; ax31 < 4; ++ax31) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + ((ic_outer * 1024) + (ax31 * 256)) / 8), 32);
      }
      __syncthreads();
      #pragma unroll
      for (int ax2 = 0; ax2 < 16; ++ax2) {
        ((int*)placeholder_shared)[((((ax2 * 128) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 16384) + (ax2 * 1024)) + (ic_outer * 128)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))];
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 4; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 4; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 4096) + (ax21 * 1024)) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 4; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    #pragma unroll
    for (int o_inner = 0; o_inner < 4; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)compute_shared + ((((int)threadIdx.z) * 256) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((unsigned int*)T_cast)[((((((((((int)blockIdx.z) * 512) + (ax0_inner_ax1_fused_inner * 256)) + (ax2_inner_ax3_inner_fused_outer * 128)) + ((((int)threadIdx.x) >> 3) * 32)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.z) * 4)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)compute_shared)[(((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)compute_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)compute_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)compute_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)compute_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)compute_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)compute_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)compute_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_pad_1_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    ((int*)T_pad)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 15) * 4096) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 15) * 256)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) / 8)))] = (((((32768 <= (((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner)) && ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 491520)) && (1 <= (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 15))) && ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 15) < 15)) ? ((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 15) * 3584) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 15) * 256)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) / 8)) - 3840))] : 0);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__kernel1(unsigned int* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  #pragma unroll
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ax3 = 0; ax3 < 16; ++ax3) {
      if (((int)threadIdx.z) < 1) {
        if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
          ((int*)compute_shared)[((((((int)threadIdx.z) * 512) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)compute)[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 512)) + (ax0_inner_ax1_fused_inner * 512)) + (ax3 * 32)) + ((int)threadIdx.x)))];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
      #pragma unroll
      for (int ax31 = 0; ax31 < 4; ++ax31) {
        if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + ((ic_outer * 1024) + (ax31 * 256)) / 8), 32);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax2 = 0; ax2 < 16; ++ax2) {
        ((int*)placeholder_shared)[((((ax2 * 128) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((int)blockIdx.y) * 8192) + (ax2 * 512)) + (ic_outer * 128)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))];
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 4; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 4; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 4096) + (ax21 * 1024)) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 4; ++o_c) {
          if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 4; ++o_inner) {
      if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
        (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 256) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
          ((int*)T_relu)[(((((((((((int)blockIdx.z) * 65536) + (ax0_inner_ax1_fused_inner * 16384)) + (ax2_inner_ax3_inner_fused_outer * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int)(((((long)((int*)placeholder2)[(((((((((((int)blockIdx.z) * 65536) + (ax0_inner_ax1_fused_inner * 16384)) + (ax2_inner_ax3_inner_fused_outer * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__6_kernel0(unsigned int* __restrict__ compute, void* __restrict__ placeholder) {
  for (int h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner = 0; h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 1; ++h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) {
    if ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) < 921600) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 115200) {
        ((int*)compute)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 30720) * 3840) + ((((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 10)) % 30) * 128)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 1023) / 8)))] = ((int*)placeholder)[((((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 30720) * 3840) + ((((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 10)) % 30) * 128)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 255) >> 5) * 16)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 1023) >> 8) * 4)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 31) / 8)))];
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[16];
  __shared__ unsigned int compute_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  __shared__ int Conv[1024];
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 16; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ax3 = 0; ax3 < 8; ++ax3) {
      #pragma unroll
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
        ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) / 28) * 28672) + ((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) % 28) * 512)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 32)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_inner = 0; ic_inner < 8; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + (ic_inner * 256) / 8), 32);
      __syncthreads();
      #pragma unroll
      for (int ax2 = 0; ax2 < 16; ++ax2) {
        ((int*)placeholder_shared)[(((ax2 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((((int)blockIdx.y) * 4096) + (ax2 * 256)) + (ic_inner * 32)) + ((int)threadIdx.x)))];
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 16; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
      }
      #pragma unroll
      for (int o_c = 0; o_c < 16; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
    #pragma unroll
    for (int o_inner = 0; o_inner < 16; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 16; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((int*)T_cast)[((((((((((int)blockIdx.z) * 16384) + (ax0_inner_ax1_fused_inner * 4096)) + (ax2_inner_ax3_inner_fused_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = (Conv[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]);
      }
    }
  }
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_3_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 6422528) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 802816) {
          ((unsigned int*)T_cast)[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 14336) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 256)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) / 8)))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 1))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 2))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 3))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 4))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 5))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 6))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1024) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) % 56) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 7))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[16];
  __shared__ unsigned int compute_shared[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 16; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 64; ++ic_outer) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
      ((int*)compute_shared)[(((int)threadIdx.x))] = ((int*)placeholder)[(((((((int)blockIdx.z) * 2048) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 256)) + (ic_outer * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
    }
    __syncthreads();
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + 0 / 8), 32);
    for (int ax2 = 0; ax2 < 16; ++ax2) {
      ((int*)placeholder_shared)[(((ax2 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((((int)blockIdx.y) * 32768) + (ax2 * 2048)) + (ic_outer * 32)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 16; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
    }
    for (int o_c = 0; o_c < 16; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  for (int o_inner = 0; o_inner < 16; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 16; ++ax3_outer_inner) {
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((unsigned int*)T_cast)[((((((((int)blockIdx.z) * 512) + (ax2_inner_ax3_inner_fused_outer * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)blockIdx.y) * 16)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[(((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[2];
  __shared__ unsigned int compute_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[128];
  #pragma unroll
  for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  #pragma unroll
  for (int ax3 = 0; ax3 < 2; ++ax3) {
    #pragma unroll
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
      ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((int)blockIdx.z) * 64) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 8)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + (ic_outer * 256) / 8), 32);
    __syncthreads();
    #pragma unroll
    for (int ax2 = 0; ax2 < 2; ++ax2) {
      ((int*)placeholder_shared)[(((ax2 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (ax2 * 64)) + (ic_outer * 32)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    #pragma unroll
    for (int ax21 = 0; ax21 < 2; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
    }
    #pragma unroll
    for (int o_c = 0; o_c < 2; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  #pragma unroll
  for (int o_inner = 0; o_inner < 2; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
    #pragma unroll
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[(((((((((int)blockIdx.z) * 2048) + (ax2_inner_ax3_inner_fused_outer * 1024)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)blockIdx.y) * 16)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((Conv[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 16) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int)(((((long)((int*)placeholder3)[(((((((((int)blockIdx.z) * 2048) + (ax2_inner_ax3_inner_fused_outer * 1024)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)blockIdx.y) * 16)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[32];
  #pragma unroll
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        #pragma unroll
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
          ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((int)blockIdx.z) * 1024) + (ax0_inner_ax1_fused_inner * 256)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 32)) + (ic_outer * 16)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax31 = 0; ax31 < 4; ++ax31) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        #pragma unroll
        for (int ax3_inner_inner = 0; ax3_inner_inner < 4; ++ax3_inner_inner) {
          ((int*)placeholder_shared)[((((ax2 * 128) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 2048) + (ax2 * 256)) + (ic_outer * 128)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 4; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + ((ax21 * 1024) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 8; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((int*)T_relu)[((((((((((int)blockIdx.z) * 32768) + (ax0_inner_ax1_fused_inner * 8192)) + (ax2_inner_ax3_inner_fused_outer * 4096)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((((int*)placeholder_shared)[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int)(((((long)((int*)placeholder3)[((((((((((int)blockIdx.z) * 32768) + (ax0_inner_ax1_fused_inner * 8192)) + (ax2_inner_ax3_inner_fused_outer * 4096)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__8_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[32];
  #pragma unroll
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 2; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        #pragma unroll
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
          if (((int)threadIdx.z) < 1) {
            ((int*)compute_shared)[((((((int)threadIdx.z) * 128) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((((((int)blockIdx.z) * 2) + ax0_inner_ax1_fused_inner) / 28) * 28672) + ((((((int)blockIdx.z) * 2) + ax0_inner_ax1_fused_inner) % 28) * 512)) + (((int)threadIdx.z) * 256)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 32)) + (ic_outer * 16)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax31 = 0; ax31 < 4; ++ax31) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 16; ++ax2) {
        #pragma unroll
        for (int ax3_inner_inner = 0; ax3_inner_inner < 2; ++ax3_inner_inner) {
          ((int*)placeholder_shared)[(((((ax2 * 128) + (((int)threadIdx.z) * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((ax2 * 256) + (ic_outer * 128)) + (((int)threadIdx.z) * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 4; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 8192) + (ax21 * 1024)) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 8; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 512) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((unsigned int*)T_cast)[(((((((((int)blockIdx.z) * 256) + (ax0_inner_ax1_fused_inner * 128)) + (ax2_inner_ax3_inner_fused_outer * 64)) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.z) * 8)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[(((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[((((((int)threadIdx.z) * 64) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_pad_4_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 1269600) {
      ((signed char*)T_pad)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((2070 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 158700)) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 158700) < 156630)) && (9 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 690))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 690) < 681)) ? ((signed char*)placeholder)[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) / 158700) * 150528) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 158700) / 690) * 672)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 690)) - 2025))] : (signed char)0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[2];
  __shared__ int placeholder_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    for (int ax3 = 0; ax3 < 2; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
        ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((int)blockIdx.z) * 128) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 16)) + (ic_outer * 8)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
      }
    }
    __syncthreads();
    for (int ax31 = 0; ax31 < 2; ++ax31) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
    }
    for (int ax2 = 0; ax2 < 4; ++ax2) {
      for (int ax3_inner_inner = 0; ax3_inner_inner < 2; ++ax3_inner_inner) {
        for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
          ((int*)placeholder_shared)[((((ax2 * 64) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 512) + (ax2 * 128)) + (ic_outer * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
        }
      }
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 4; ++ax21) {
      for (int ax32 = 0; ax32 < 2; ++ax32) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 2) + ax32)], ((int *)placeholder_shared + ((ax21 * 512) + (ax32 * 256)) / 8), 32);
      }
    }
    for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
      for (int o_c = 0; o_c < 4; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 2) + ic_inner)], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  __syncthreads();
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[(((((((((int)blockIdx.z) * 4096) + (ax2_inner_ax3_inner_fused_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((((int*)placeholder_shared)[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int)(((((long)((int*)placeholder3)[(((((((((int)blockIdx.z) * 4096) + (ax2_inner_ax3_inner_fused_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[2];
  __shared__ unsigned int compute_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[128];
  #pragma unroll
  for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  #pragma unroll
  for (int ax3 = 0; ax3 < 2; ++ax3) {
    #pragma unroll
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
      ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((int)blockIdx.z) * 64) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 8)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + (ic_outer * 256) / 8), 32);
    __syncthreads();
    #pragma unroll
    for (int ax2 = 0; ax2 < 2; ++ax2) {
      ((int*)placeholder_shared)[(((ax2 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (ax2 * 64)) + (ic_outer * 32)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    #pragma unroll
    for (int ax21 = 0; ax21 < 2; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
    }
    #pragma unroll
    for (int o_c = 0; o_c < 2; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  #pragma unroll
  for (int o_inner = 0; o_inner < 2; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
    #pragma unroll
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[(((((((((int)blockIdx.z) * 2048) + (ax2_inner_ax3_inner_fused_outer * 1024)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)blockIdx.y) * 16)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((Conv[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 16) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int*)placeholder3)[(((((((((int)blockIdx.z) * 2048) + (ax2_inner_ax3_inner_fused_outer * 1024)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)blockIdx.y) * 16)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__kernel0(unsigned int* __restrict__ compute, void* __restrict__ placeholder) {
  for (int h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner = 0; h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 1; ++h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) {
    if ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) < 200704) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 25088) {
        ((int*)compute)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 28672) * 3584) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 12)) % 7) * 512)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 4095) / 8)))] = ((int*)placeholder)[((((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 28672) * 3584) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 12)) % 7) * 512)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 255) >> 5) * 64)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 4095) >> 8) * 4)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 31) / 8)))];
      }
    }
  }
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    ((unsigned int*)T_cast)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 2048) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 256)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) / 8)))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 1))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 2))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 3))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 4))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 5))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 6))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 14) * 16384) + ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 11)) & 7) * 2048)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 2047) >> 3) * 8)) + 7))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
  }
}

extern "C" __global__ void fused_nn_pad_3_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    if ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 1722368) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 215296) {
        ((int*)T_pad)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 29696) * 3712) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 58) * 64)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) / 8)))] = (((((29696 <= (((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner)) && ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 1692672)) && (1 <= (((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 58))) && ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 58) < 57)) ? ((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 29696) * 3584) + ((((((int)blockIdx.x) * 16) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 9)) % 58) * 64)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 511) / 8)) - 3648))] : 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__11_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[512];
  #pragma unroll
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  #pragma unroll
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    #pragma unroll
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
      if (((int)threadIdx.y) < 1) {
        ((int*)compute_shared)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((int)blockIdx.z) * 64) + (((int)threadIdx.y) * 64)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 8)) + (ic_outer * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
      }
    }
    __syncthreads();
    if (((int)threadIdx.y) < 1) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + (((int)threadIdx.y) * 256) / 8), 32);
    }
    #pragma unroll
    for (int ax2 = 0; ax2 < 8; ++ax2) {
      #pragma unroll
      for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
        if (((int)threadIdx.y) < 1) {
          ((int*)placeholder_shared)[((((ax2 * 32) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((ax2 * 64) + (ic_outer * 32)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ax21 = 0; ax21 < 8; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
    }
    #pragma unroll
    for (int o_c = 0; o_c < 8; ++o_c) {
      if (((int)threadIdx.y) < 1) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  #pragma unroll
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    if (((int)threadIdx.y) < 1) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + ((((int)threadIdx.y) * 512) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    #pragma unroll
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      if ((((((int)threadIdx.y) * 8) + (ax2_inner_ax3_inner_fused_outer * 4)) + (((int)threadIdx.x) >> 3)) < 8) {
        if (((int)threadIdx.y) < 1) {
          ((unsigned int*)T_cast)[((((((((int)blockIdx.z) * 64) + (((int)threadIdx.y) * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((Conv[(((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[((ax3_outer_inner * 8))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((Conv[((((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((Conv[((((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((Conv[((((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((Conv[((((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((Conv[((((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((Conv[((((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((Conv[((((((((int)threadIdx.y) * 512) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_dense_add_cast_subtract_cast_multiply_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_multiply, void* __restrict__ placeholder2) {
  int compute_local[5];
  __shared__ signed char placeholder_shared[2048];
  __shared__ signed char placeholder_shared1[2560];
  compute_local[(0)] = 0;
  compute_local[(1)] = 0;
  compute_local[(2)] = 0;
  compute_local[(3)] = 0;
  compute_local[(4)] = 0;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_outer_fused_outer_outer = 0; ax0_ax1_outer_fused_outer_outer < 8; ++ax0_ax1_outer_fused_outer_outer) {
      ((int4*)(placeholder_shared + ((((ax0_ax1_outer_fused_outer_outer * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + (((((ax0_ax1_outer_fused_outer_outer * 2048) + (k_outer_outer * 256)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 16)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_outer_fused_outer_outer1 = 0; ax0_ax1_outer_fused_outer_outer1 < 10; ++ax0_ax1_outer_fused_outer_outer1) {
      ((int4*)(placeholder_shared1 + ((((ax0_ax1_outer_fused_outer_outer1 * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + ((((((((int)blockIdx.x) * 20480) + (ax0_ax1_outer_fused_outer_outer1 * 2048)) + (k_outer_outer * 256)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 16)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int k_outer_inner = 0; k_outer_inner < 64; ++k_outer_inner) {
      compute_local[(0)] = __dp4a(((int*)(placeholder_shared + (((((int)threadIdx.y) * 256) + (k_outer_inner * 4)))))[0], ((int*)(placeholder_shared1 + (((((int)threadIdx.x) * 256) + (k_outer_inner * 4)))))[0], compute_local[(0)]);
      compute_local[(1)] = __dp4a(((int*)(placeholder_shared + (((((int)threadIdx.y) * 256) + (k_outer_inner * 4)))))[0], ((int*)(placeholder_shared1 + ((((((int)threadIdx.x) * 256) + (k_outer_inner * 4)) + 512))))[0], compute_local[(1)]);
      compute_local[(2)] = __dp4a(((int*)(placeholder_shared + (((((int)threadIdx.y) * 256) + (k_outer_inner * 4)))))[0], ((int*)(placeholder_shared1 + ((((((int)threadIdx.x) * 256) + (k_outer_inner * 4)) + 1024))))[0], compute_local[(2)]);
      compute_local[(3)] = __dp4a(((int*)(placeholder_shared + (((((int)threadIdx.y) * 256) + (k_outer_inner * 4)))))[0], ((int*)(placeholder_shared1 + ((((((int)threadIdx.x) * 256) + (k_outer_inner * 4)) + 1536))))[0], compute_local[(3)]);
      compute_local[(4)] = __dp4a(((int*)(placeholder_shared + (((((int)threadIdx.y) * 256) + (k_outer_inner * 4)))))[0], ((int*)(placeholder_shared1 + ((((((int)threadIdx.x) * 256) + (k_outer_inner * 4)) + 2048))))[0], compute_local[(4)]);
    }
  }
  ((float*)T_multiply)[((((((int)threadIdx.y) * 1000) + (((int)blockIdx.x) * 10)) + ((int)threadIdx.x)))] = (((float)(compute_local[(0)] + ((int*)placeholder2)[(((((int)blockIdx.x) * 10) + ((int)threadIdx.x)))])) * 7.400000e+01f);
  ((float*)T_multiply)[(((((((int)threadIdx.y) * 1000) + (((int)blockIdx.x) * 10)) + ((int)threadIdx.x)) + 2))] = (((float)(compute_local[(1)] + ((int*)placeholder2)[((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) + 2))])) * 7.400000e+01f);
  ((float*)T_multiply)[(((((((int)threadIdx.y) * 1000) + (((int)blockIdx.x) * 10)) + ((int)threadIdx.x)) + 4))] = (((float)(compute_local[(2)] + ((int*)placeholder2)[((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) + 4))])) * 7.400000e+01f);
  ((float*)T_multiply)[(((((((int)threadIdx.y) * 1000) + (((int)blockIdx.x) * 10)) + ((int)threadIdx.x)) + 6))] = (((float)(compute_local[(3)] + ((int*)placeholder2)[((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) + 6))])) * 7.400000e+01f);
  ((float*)T_multiply)[(((((((int)threadIdx.y) * 1000) + (((int)blockIdx.x) * 10)) + ((int)threadIdx.x)) + 8))] = (((float)(compute_local[(4)] + ((int*)placeholder2)[((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) + 8))])) * 7.400000e+01f);
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_1_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    ((unsigned int*)T_cast)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 14336) + ((((int)blockIdx.x) % 14) * 1024)) + ((int)threadIdx.x)))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)) + 1))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)) + 2))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)) + 3))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)) + 4))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)) + 5))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)) + 6))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + ((((int)blockIdx.x) % 14) * 8192)) + (((int)threadIdx.x) * 8)) + 7))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
  }
}

extern "C" __global__ void fused_nn_pad_2_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
    if ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 921600) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 115200) {
        ((int*)T_pad)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 30720) * 3840) + ((((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 10)) % 30) * 128)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 1023) / 8)))] = (((((30720 <= (((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner)) && ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 890880)) && (1 <= (((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 10)) % 30))) && ((((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 10)) % 30) < 29)) ? ((int*)placeholder)[(((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 30720) * 3584) + ((((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 10)) % 30) * 128)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 1023) / 8)) - 3712))] : 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[2];
  __shared__ unsigned int compute_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[128];
  #pragma unroll
  for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  #pragma unroll
  for (int ax3 = 0; ax3 < 2; ++ax3) {
    #pragma unroll
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
      ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((int)blockIdx.z) * 64) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 8)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + (ic_outer * 256) / 8), 32);
    __syncthreads();
    #pragma unroll
    for (int ax2 = 0; ax2 < 2; ++ax2) {
      ((int*)placeholder_shared)[(((ax2 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (ax2 * 64)) + (ic_outer * 32)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    #pragma unroll
    for (int ax21 = 0; ax21 < 2; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
    }
    #pragma unroll
    for (int o_c = 0; o_c < 2; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  #pragma unroll
  for (int o_inner = 0; o_inner < 2; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
    #pragma unroll
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((int*)T_cast)[(((((((((int)blockIdx.z) * 2048) + (ax2_inner_ax3_inner_fused_outer * 1024)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)blockIdx.y) * 16)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = (Conv[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 16) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[16];
  __shared__ unsigned int compute_shared[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ int placeholder_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 16; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 32; ++ic_outer) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
      ((int*)compute_shared)[(((int)threadIdx.x))] = ((int*)placeholder)[(((((((((int)blockIdx.z) / 7) * 28672) + ((((int)blockIdx.z) % 7) * 2048)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 128)) + (ic_outer * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
    }
    __syncthreads();
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((unsigned int *)compute_shared + 0 / 8), 32);
    for (int ax2 = 0; ax2 < 16; ++ax2) {
      for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
        ((int*)placeholder_shared)[(((ax2 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder1)[(((((((int)blockIdx.y) * 16384) + (ax2 * 1024)) + (ic_outer * 32)) + ((int)threadIdx.x)))];
      }
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 16; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((int *)placeholder_shared + (ax21 * 256) / 8), 32);
    }
    for (int o_c = 0; o_c < 16; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  for (int o_inner = 0; o_inner < 16; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 16; ++ax3_outer_inner) {
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((unsigned int*)T_cast)[((((((((int)blockIdx.z) * 512) + (ax2_inner_ax3_inner_fused_outer * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)blockIdx.y) * 16)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[(((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((Conv[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[768];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  #pragma unroll
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int kh = 0; kh < 3; ++kh) {
    #pragma unroll
    for (int ax1 = 0; ax1 < 3; ++ax1) {
      #pragma unroll
      for (int ax3 = 0; ax3 < 8; ++ax3) {
        #pragma unroll
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
          if (((int)threadIdx.z) < 1) {
            ((int*)compute_shared)[(((((ax1 * 256) + (((int)threadIdx.z) * 256)) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((((((int)blockIdx.z) / 14) * 4096) + (kh * 4096)) + (ax1 * 256)) + (((int)threadIdx.z) * 256)) + ((((int)blockIdx.z) % 14) * 256)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 32)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      #pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        #pragma unroll
        for (int ax31 = 0; ax31 < 4; ++ax31) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (((kw * 2048) + (ic_outer * 1024)) + (ax31 * 256)) / 8), 32);
        }
        __syncthreads();
        #pragma unroll
        for (int ax2 = 0; ax2 < 16; ++ax2) {
          ((int*)placeholder_shared)[((((ax2 * 128) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((kh * 24576) + (kw * 8192)) + (((int)blockIdx.y) * 4096)) + (ax2 * 256)) + (ic_outer * 128)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))];
        }
        __syncthreads();
        #pragma unroll
        for (int ax21 = 0; ax21 < 4; ++ax21) {
          #pragma unroll
          for (int ax32 = 0; ax32 < 4; ++ax32) {
            (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 4096) + (ax21 * 1024)) + (ax32 * 256)) / 8), 32);
          }
        }
        #pragma unroll
        for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
          #pragma unroll
          for (int o_c = 0; o_c < 4; ++o_c) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 256) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    #pragma unroll
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((unsigned int*)T_cast)[(((((((((int)blockIdx.z) * 256) + (ax2_inner_ax3_inner_fused_outer * 128)) + ((((int)threadIdx.x) >> 3) * 32)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.z) * 4)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__6_kernel1(unsigned int* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_cast, void* __restrict__ placeholder1) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[384];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 3; ++ax1) {
        #pragma unroll
        for (int ax3 = 0; ax3 < 4; ++ax3) {
          if (((int)threadIdx.z) < 1) {
            ((int*)compute_shared)[(((((ax1 * 128) + (((int)threadIdx.z) * 128)) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)compute)[(((((((((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) / 28) * 3840) + (kh * 3840)) + (ax1 * 128)) + (((int)threadIdx.z) * 128)) + ((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) % 28) * 128)) + (ax3 * 32)) + ((int)threadIdx.x)))];
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        #pragma unroll
        for (int ax31 = 0; ax31 < 4; ++ax31) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + ((kw * 1024) + (ax31 * 256)) / 8), 32);
        }
        __syncthreads();
        #pragma unroll
        for (int ax2 = 0; ax2 < 8; ++ax2) {
          #pragma unroll
          for (int ax3_inner_inner = 0; ax3_inner_inner < 2; ++ax3_inner_inner) {
            ((int*)placeholder_shared)[(((((ax2 * 128) + (((int)threadIdx.z) * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((kh * 6144) + (kw * 2048)) + (((int)blockIdx.y) * 1024)) + (ax2 * 128)) + (((int)threadIdx.z) * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
          }
        }
        __syncthreads();
        #pragma unroll
        for (int ax21 = 0; ax21 < 4; ++ax21) {
          #pragma unroll
          for (int ax32 = 0; ax32 < 4; ++ax32) {
            (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 4096) + (ax21 * 1024)) + (ax32 * 256)) / 8), 32);
          }
        }
        #pragma unroll
        for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
          #pragma unroll
          for (int o_c = 0; o_c < 4; ++o_c) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 4; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 256) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((unsigned int*)T_cast)[((((((((((int)blockIdx.z) * 512) + (ax0_inner_ax1_fused_inner * 128)) + (ax2_inner_ax3_inner_fused_outer * 64)) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.z) * 4)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[((((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_kernel1(unsigned int* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[4];
  __shared__ int placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  #pragma unroll
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int ax3 = 0; ax3 < 16; ++ax3) {
      if (((int)threadIdx.z) < 1) {
        if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
          ((int*)compute_shared)[((((((int)threadIdx.z) * 512) + (ax3 * 32)) + ((int)threadIdx.x)))] = ((int*)compute)[((((((((int)blockIdx.z) * 2048) + (((int)threadIdx.z) * 512)) + (ax0_inner_ax1_fused_inner * 512)) + (ax3 * 32)) + ((int)threadIdx.x)))];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
      #pragma unroll
      for (int ax31 = 0; ax31 < 4; ++ax31) {
        if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + ((ic_outer * 1024) + (ax31 * 256)) / 8), 32);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ax2 = 0; ax2 < 16; ++ax2) {
        ((int*)placeholder_shared)[((((ax2 * 128) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((int)blockIdx.y) * 8192) + (ax2 * 512)) + (ic_outer * 128)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))];
      }
      __syncthreads();
      #pragma unroll
      for (int ax21 = 0; ax21 < 4; ++ax21) {
        #pragma unroll
        for (int ax32 = 0; ax32 < 4; ++ax32) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 4) + ax32)], ((int *)placeholder_shared + (((((int)threadIdx.z) * 4096) + (ax21 * 1024)) + (ax32 * 256)) / 8), 32);
        }
      }
      #pragma unroll
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        #pragma unroll
        for (int o_c = 0; o_c < 4; ++o_c) {
          if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 4) + ic_inner)], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 4; ++o_inner) {
      if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
        (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + ((((int)threadIdx.z) * 256) + (o_inner * 64))), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        if (((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) < 49) {
          ((int*)T_relu)[(((((((((((int)blockIdx.z) * 65536) + (ax0_inner_ax1_fused_inner * 16384)) + (ax2_inner_ax3_inner_fused_outer * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((((int*)placeholder_shared)[(((((((int)threadIdx.z) * 256) + (ax3_outer_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int*)placeholder2)[(((((((((((int)blockIdx.z) * 65536) + (ax0_inner_ax1_fused_inner * 16384)) + (ax2_inner_ax3_inner_fused_outer * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.z) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]), 2147483647), 0);
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_kernel0(unsigned int* __restrict__ compute, void* __restrict__ placeholder) {
  for (int h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner = 0; h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 1; ++h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) {
    if ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) < 200704) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 25088) {
        ((int*)compute)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 28672) * 3584) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 12)) % 7) * 512)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 4095) / 8)))] = ((int*)placeholder)[((((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 28672) * 3584) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 12)) % 7) * 512)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 255) >> 5) * 64)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 4095) >> 8) * 4)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 31) / 8)))];
      }
    }
  }
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_2_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) < 3211264) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 401408) {
          ((unsigned int*)T_cast)[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 14336) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 512)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) / 8)))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)((int*)placeholder)[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)) + 1))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)) + 2))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)) + 3))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)) + 4))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)) + 5))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)) + 6))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)((int*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 2097152) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.x) * 8)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) / 114688) * 114688) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) >> 12)) % 28) * 4096)) + (((((((int)threadIdx.x) * 8) + ax0_ax1_fused_ax2_fused_ax3_fused_inner) & 4095) >> 3) * 8)) + 7))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_batch_flatten_kernel0(void* __restrict__ tensor, void* __restrict__ placeholder) {
  ((signed char*)tensor)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))];
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[4];
  __shared__ unsigned int compute_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[2];
  __shared__ int placeholder_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    for (int ax3 = 0; ax3 < 2; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
        ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[((((((((int)blockIdx.z) * 128) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 16)) + (ic_outer * 8)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
      }
    }
    __syncthreads();
    for (int ax31 = 0; ax31 < 2; ++ax31) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
    }
    for (int ax2 = 0; ax2 < 4; ++ax2) {
      for (int ax3_inner_inner = 0; ax3_inner_inner < 2; ++ax3_inner_inner) {
        for (int ax4_ax5_fused_inner_inner = 0; ax4_ax5_fused_inner_inner < 8; ++ax4_ax5_fused_inner_inner) {
          ((int*)placeholder_shared)[((((ax2 * 64) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((((int)blockIdx.y) * 512) + (ax2 * 128)) + (ic_outer * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
        }
      }
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 4; ++ax21) {
      for (int ax32 = 0; ax32 < 2; ++ax32) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 2) + ax32)], ((int *)placeholder_shared + ((ax21 * 512) + (ax32 * 256)) / 8), 32);
      }
    }
    for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
      for (int o_c = 0; o_c < 4; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 2) + ic_inner)], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  __syncthreads();
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[(((((((((int)blockIdx.z) * 4096) + (ax2_inner_ax3_inner_fused_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))] = max(min(((((int*)placeholder_shared)[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]) + ((int*)placeholder3)[(((((((((int)blockIdx.z) * 4096) + (ax2_inner_ax3_inner_fused_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 32)) + (ax3_outer_inner * 8)) + (((int)threadIdx.x) & 7)))]), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_multiply_add_right_shift_cast_nn_relu_kernel2(void* __restrict__ T_relu, int* __restrict__ gemm_C, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) < 100352) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((int*)T_relu)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = max(((int)(((((long)(gemm_C[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + ((((int)threadIdx.x) >> 6) * 16)) + (((int)threadIdx.x) & 15)))] + ((int*)placeholder)[((((int)threadIdx.x) & 63))])) * (long)1857283155) + (long)1073741824) >> (long)31)), 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__kernel0(unsigned int* __restrict__ compute, void* __restrict__ placeholder) {
  for (int h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner = 0; h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 1; ++h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) {
    if ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) < 331776) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 41472) {
        ((int*)compute)[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 36864) * 4608) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 12)) % 9) * 512)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 4095) / 8)))] = ((int*)placeholder)[((((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 8)) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) / 36864) * 4608) + ((((((int)blockIdx.x) * 2) + (((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) >> 12)) % 9) * 512)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 255) >> 5) * 64)) + (((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 4095) >> 8) * 4)) + ((((((int)threadIdx.x) * 8) + h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner) & 31) / 8)))];
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__9_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 32, int> Conv_wmma_accumulator[8];
  __shared__ unsigned int compute_shared[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 32, nvcuda::wmma::experimental::precision::u4, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[2];
  __shared__ int placeholder_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 32, nvcuda::wmma::experimental::precision::s4, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[16];
  for (int ax0_inner_ax1_fused_inner = 0; ax0_inner_ax1_fused_inner < 4; ++ax0_inner_ax1_fused_inner) {
    #pragma unroll
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        #pragma unroll
        for (int ax3 = 0; ax3 < 2; ++ax3) {
          #pragma unroll
          for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 8; ++ax4_ax5_fused_inner_outer) {
            ((int*)compute_shared)[(((ax3 * 32) + ((int)threadIdx.x)))] = ((int*)placeholder)[(((((((((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) / 56) * 3712) + (kh * 3712)) + (kw * 64)) + ((((((int)blockIdx.z) * 4) + ax0_inner_ax1_fused_inner) % 56) * 64)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) >> 5) * 8)) + (ax3 * 4)) + ((((((int)threadIdx.x) * 8) + ax4_ax5_fused_inner_outer) & 31) / 8)))];
          }
        }
        __syncthreads();
        #pragma unroll
        for (int ax31 = 0; ax31 < 2; ++ax31) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[ax31], ((unsigned int *)compute_shared + (ax31 * 256) / 8), 32);
        }
        #pragma unroll
        for (int ax2 = 0; ax2 < 8; ++ax2) {
          #pragma unroll
          for (int ax3_inner_inner = 0; ax3_inner_inner < 2; ++ax3_inner_inner) {
            ((int*)placeholder_shared)[((((ax2 * 64) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))] = ((int*)placeholder1)[((((((kh * 1536) + (kw * 512)) + (ax2 * 64)) + (ax3_inner_inner * 32)) + ((int)threadIdx.x)))];
          }
        }
        __syncthreads();
        #pragma unroll
        for (int ax21 = 0; ax21 < 8; ++ax21) {
          #pragma unroll
          for (int ax32 = 0; ax32 < 2; ++ax32) {
            (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[((ax21 * 2) + ax32)], ((int *)placeholder_shared + ((ax21 * 512) + (ax32 * 256)) / 8), 32);
          }
        }
        #pragma unroll
        for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
          #pragma unroll
          for (int o_c = 0; o_c < 8; ++o_c) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[ic_inner], placeholder_shared_wmma_matrix_b[((o_c * 2) + ic_inner)], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)placeholder_shared + (o_inner * 64)), Conv_wmma_accumulator[o_inner], 8, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    #pragma unroll
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      #pragma unroll
      for (int ax2_inner_ax3_inner_fused_outer = 0; ax2_inner_ax3_inner_fused_outer < 2; ++ax2_inner_ax3_inner_fused_outer) {
        ((unsigned int*)T_cast)[((((((((int)blockIdx.z) * 256) + (ax0_inner_ax1_fused_inner * 64)) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)))] + ((int*)placeholder2)[((ax3_outer_inner * 8))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 1))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 2))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 3))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 4))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 5))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 6))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (max(min(((int)((((((long)max((((int*)placeholder_shared)[(((((ax3_outer_inner * 64) + (ax2_inner_ax3_inner_fused_outer * 32)) + ((((int)threadIdx.x) >> 3) * 8)) + 7))] + ((int*)placeholder2)[(((ax3_outer_inner * 8) + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));
      }
    }
  }
}

