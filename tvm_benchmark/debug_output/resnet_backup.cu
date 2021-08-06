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

extern "C" __global__ void fused_nn_pad_4_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 1269600) {
      ((signed char*)T_pad)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((2070 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 158700)) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 158700) < 156630)) && (9 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 690))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 690) < 681)) ? ((signed char*)placeholder)[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) / 158700) * 150528) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 158700) / 690) * 672)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 690)) - 2025))] : (signed char)0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_multiply_add_right_shift_cast_nn_relu_kernel0(signed char* __restrict__ kernel_im2col_pack, void* __restrict__ placeholder) {
  kernel_im2col_pack[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = (((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 10) * 16) + (((int)threadIdx.x) & 15)) < 147) ? ((signed char*)placeholder)[(((((((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 10) * 16) + (((int)threadIdx.x) & 15)) / 3) * 192) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 10) * 48)) + (((((int)threadIdx.x) & 255) >> 4) * 3)) + ((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 10) * 16) + (((int)threadIdx.x) & 15)) % 3)))] : (signed char)0);
}

extern "C" __global__ void fused_nn_conv2d_add_cast_3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 16; ++ic_outer) {
    for (int ax3 = 0; ax3 < 4; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 200704) + ((((int)blockIdx.z) / 7) * 28672)) + ((((int)blockIdx.z) % 7) * 2048)) + (ic_outer * 64)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 262144) + (ax2 * 32768)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 8; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = (Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ax3 = 0; ax3 < 4; ++ax3) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
      compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 200704) + (((int)blockIdx.z) * 64)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
    }
  }
  __syncthreads();
  for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
    __syncthreads();
    for (int ax2 = 0; ax2 < 4; ++ax2) {
      ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (ax2 * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 4; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
    }
    for (int o_c = 0; o_c < 4; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[((((((ax0_inner_ax3_inner_fused_outer * 802816) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int)(((((long)((int*)placeholder3)[((((((ax0_inner_ax3_inner_fused_outer * 802816) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_3_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 1024) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) >> 8)) < 25088) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 127), -128));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__10_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[2];
  __shared__ signed char compute_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[512];
  for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    for (int ax3 = 0; ax3 < 8; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[(((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 802816) + (((int)blockIdx.z) * 256)) + (ic_outer * 128)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 8; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 2; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + (((((ax2 * 8192) + (ic_outer * 4096)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 2; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 2; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 2; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 64)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
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

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__9_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 7; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) < 3364) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 13456) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 107648) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 1722368) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) & 127) >> 4) * 215296) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 7) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  compute[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((((((int)threadIdx.x) & 127) >> 4) * 25088) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 7) * 16)) + (((int)threadIdx.x) & 15)))];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__7_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
    for (int ax3 = 0; ax3 < 8; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[(((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 401408) + (((int)blockIdx.z) * 512)) + (ic_outer * 128)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 8; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 4; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + (((((ax2 * 16384) + (ic_outer * 4096)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 4; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 4; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__8_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 8; ++ic_outer) {
    for (int ax3 = 0; ax3 < 2; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 802816) + ((((int)blockIdx.z) / 28) * 28672)) + ((((int)blockIdx.z) % 28) * 512)) + (ic_outer * 32)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 4; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + (((((ax2 * 8192) + (ic_outer * 1024)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 4; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 4; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 2; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if (((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 64) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) >> 5)) < 81) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 2592) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 20736) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 331776) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((((((int)threadIdx.x) & 127) >> 4) * 41472) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__4_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 64; ++ic_outer) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
      compute_shared[(((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 200704) + (((int)blockIdx.z) * 1024)) + (ic_outer * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
    }
    __syncthreads();
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + 0), 16);
    for (int ax2 = 0; ax2 < 8; ++ax2) {
      ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + ((((ax2 * 32768) + (ic_outer * 512)) + (((int)threadIdx.x) * 16)))))[0];
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 8; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
    }
    for (int o_c = 0; o_c < 8; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 50176) + (((int)blockIdx.z) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_pad_3_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) < 26912) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 1722368) {
        ((signed char*)T_pad)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((58 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 3364)) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 3364) < 3306)) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 58))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 58) < 57)) ? ((signed char*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) / 3364) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 3364) / 58) * 3584)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 58) * 64)) + (((int)threadIdx.x) & 63)) - 3648))] : (signed char)0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_max_pool2d_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_kernel0(void* __restrict__ placeholder, void* __restrict__ T_cast) {
  int tensor[1];
  tensor[(0)] = -2147483648;
  for (int rv = 0; rv < 3; ++rv) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor[(0)] = max(tensor[(0)], (((1 <= ((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 3136) / 56) * 2) + rv)) && (1 <= (((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 56) * 2) + rv1))) ? ((int*)placeholder)[((((((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 56) * 14336) + (rv * 7168)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 56) * 128)) + (rv1 * 64)) + (((int)threadIdx.x) & 63)) - 7232))] : -2147483648));
    }
  }
  ((signed char*)T_cast)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)tensor[(0)]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 127), -128));
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__1_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 2; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if (((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 128) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) >> 4)) < 196) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 3136) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 25088) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 401408) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((((((int)threadIdx.x) & 127) >> 4) * 50176) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__1_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_cast, void* __restrict__ placeholder1) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 16; ++ic_outer) {
    for (int ax3 = 0; ax3 < 8; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[((((((((int)blockIdx.z) * 16384) + (ic_outer * 1024)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 8; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 4; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 262144) + (ax2 * 65536)) + (ic_outer * 4096)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 4; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 4; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 25088) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__1_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 4; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if (((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) >> 7)) < 49) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 6272) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 50176) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 802816) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((((((int)threadIdx.x) & 127) >> 4) * 100352) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_2_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 4; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if (((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 256) + ((int)blockIdx.x)) < 784) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 6272) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 50176) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 802816) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) & 127) >> 4) * 100352) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 7) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_pad_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) < 648) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 331776) {
        ((signed char*)T_pad)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((9 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) % 81)) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) % 81) < 72)) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) % 9))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) % 9) < 8)) ? ((signed char*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) / 81) * 25088) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) % 81) / 9) * 3584)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) % 9) * 512)) + (((int)threadIdx.x) & 511)) - 4096))] : (signed char)0);
      }
    }
  }
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_2_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 512) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) >> 9)) < 6272) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 3211264) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 127), -128));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 8; ++ic_outer) {
    for (int ax3 = 0; ax3 < 4; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 401408) + ((((int)blockIdx.z) / 14) * 28672)) + ((((int)blockIdx.z) % 14) * 1024)) + (ic_outer * 64)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 131072) + (ax2 * 16384)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 8; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 1024)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = (Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]);
    }
  }
}

extern "C" __global__ void fused_nn_pad_1_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    ((signed char*)T_pad)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((16 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) & 255)) && ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) & 255) < 240)) && (1 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) & 15))) && ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) & 15) < 15)) ? ((signed char*)placeholder)[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 200704) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) >> 8) * 50176)) + (((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) & 255) >> 4) * 3584)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) & 15) * 256)) + (((int)threadIdx.x) & 255)) - 3840))] : (signed char)0);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__3_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_cast, void* __restrict__ placeholder1) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[768];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 8; ++ic_outer) {
    for (int kh = 0; kh < 3; ++kh) {
      for (int ax2 = 0; ax2 < 3; ++ax2) {
        for (int ax3 = 0; ax3 < 2; ++ax3) {
          for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
            compute_shared[(((((ax2 * 256) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[((((((((((((int)blockIdx.z) / 14) * 32768) + (kh * 32768)) + (ax2 * 2048)) + ((((int)blockIdx.z) % 14) * 2048)) + (ic_outer * 256)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
          }
        }
      }
      __syncthreads();
      for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
        for (int kw = 0; kw < 3; ++kw) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + ((kw * 256) + (ic_inner * 128))), 16);
          __syncthreads();
          for (int ax21 = 0; ax21 < 8; ++ax21) {
            ((int4*)(placeholder_shared + (((ax21 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + (((((((kh * 196608) + (kw * 65536)) + (ax21 * 8192)) + (ic_outer * 1024)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
          }
          __syncthreads();
          for (int ax22 = 0; ax22 < 8; ++ax22) {
            (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax22], ((signed char *)placeholder_shared + (ax22 * 512)), 16);
          }
          for (int o_c = 0; o_c < 8; ++o_c) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 50176) + (((int)blockIdx.z) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 128) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) >> 11)) < 392) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 802816) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 127), -128));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__9_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_cast, void* __restrict__ placeholder1) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[2];
  __shared__ signed char compute_shared[768];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[512];
  for (int ax1_ax2_fused_inner = 0; ax1_ax2_fused_inner < 2; ++ax1_ax2_fused_inner) {
    for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      for (int kh = 0; kh < 3; ++kh) {
        for (int ax2 = 0; ax2 < 3; ++ax2) {
          for (int ax3 = 0; ax3 < 2; ++ax3) {
            for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
              compute_shared[(((((ax2 * 256) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[((((((((((((((int)blockIdx.z) * 2) + ax1_ax2_fused_inner) / 56) * 29696) + (kh * 29696)) + (ax2 * 512)) + ((((((int)blockIdx.z) * 2) + ax1_ax2_fused_inner) % 56) * 512)) + (ic_outer * 256)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
            }
          }
        }
        __syncthreads();
        for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
          for (int kw = 0; kw < 3; ++kw) {
            (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + ((kw * 256) + (ic_inner * 128))), 16);
            __syncthreads();
            for (int ax21 = 0; ax21 < 2; ++ax21) {
              ((int4*)(placeholder_shared + (((ax21 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + (((((((kh * 12288) + (kw * 4096)) + (ax21 * 2048)) + (ic_outer * 1024)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
            }
            __syncthreads();
            for (int ax22 = 0; ax22 < 2; ++ax22) {
              (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax22], ((signed char *)placeholder_shared + (ax22 * 512)), 16);
            }
            for (int o_c = 0; o_c < 2; ++o_c) {
              (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
            }
          }
        }
      }
    }
    for (int o_inner = 0; o_inner < 2; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
      for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
        ((signed char*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 128)) + (ax1_ax2_fused_inner * 64)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__5_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 32; ++ic_outer) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
      compute_shared[(((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[(((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 401408) + ((((int)blockIdx.z) / 14) * 28672)) + ((((int)blockIdx.z) % 14) * 1024)) + (ic_outer * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
    }
    __syncthreads();
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + 0), 16);
    for (int ax2 = 0; ax2 < 8; ++ax2) {
      ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + ((((ax2 * 16384) + (ic_outer * 512)) + (((int)threadIdx.x) * 16)))))[0];
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 8; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
    }
    for (int o_c = 0; o_c < 8; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 50176) + (((int)blockIdx.z) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_1_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 2; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if (((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 128) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) >> 4)) < 196) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 3136) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 25088) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 401408) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((((((int)threadIdx.x) & 127) >> 4) * 50176) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__11_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[2];
  __shared__ signed char compute_shared[256];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[512];
  for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    for (int ax3 = 0; ax3 < 2; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[(((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 200704) + (((int)blockIdx.z) * 64)) + (ic_outer * 32)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 2; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + (((((ax2 * 2048) + (ic_outer * 1024)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 2; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 2; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 2; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 64)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_1_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int ax1_ax2_fused_inner = 0; ax1_ax2_fused_inner < 2; ++ax1_ax2_fused_inner) {
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
          compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[(((((((((int)blockIdx.z) * 4096) + (ax1_ax2_fused_inner * 2048)) + (ic_outer * 512)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
        }
      }
      __syncthreads();
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
        __syncthreads();
        for (int ax2 = 0; ax2 < 8; ++ax2) {
          ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 65536) + (ax2 * 8192)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
        }
        __syncthreads();
        for (int ax21 = 0; ax21 < 8; ++ax21) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
        }
        for (int o_c = 0; o_c < 8; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
        ((int*)T_relu)[(((((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 2048)) + (ax1_ax2_fused_inner * 1024)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int*)placeholder2)[(((((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 2048)) + (ax1_ax2_fused_inner * 1024)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]), 2147483647), 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 8; ++ic_outer) {
    for (int ax3 = 0; ax3 < 4; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[((((((((int)blockIdx.z) * 4096) + (ic_outer * 512)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 131072) + (ax2 * 16384)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 8; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[((((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int)(((((long)((int*)placeholder2)[((((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ax3 = 0; ax3 < 4; ++ax3) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
      compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 200704) + (((int)blockIdx.z) * 64)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
    }
  }
  __syncthreads();
  for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
    __syncthreads();
    for (int ax2 = 0; ax2 < 4; ++ax2) {
      ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (ax2 * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 4; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
    }
    for (int o_c = 0; o_c < 4; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[((((((ax0_inner_ax3_inner_fused_outer * 802816) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int*)placeholder3)[((((((ax0_inner_ax3_inner_fused_outer * 802816) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  compute[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((((((int)threadIdx.x) & 127) >> 4) * 25088) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 7) * 16)) + (((int)threadIdx.x) & 15)))];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__6_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_cast, void* __restrict__ placeholder1) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[1536];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
    for (int kh = 0; kh < 3; ++kh) {
      for (int ax2 = 0; ax2 < 3; ++ax2) {
        for (int ax3 = 0; ax3 < 4; ++ax3) {
          for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
            compute_shared[(((((ax2 * 512) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[((((((((((((int)blockIdx.z) / 28) * 30720) + (kh * 30720)) + (ax2 * 1024)) + ((((int)blockIdx.z) % 28) * 1024)) + (ic_outer * 512)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
          }
        }
      }
      __syncthreads();
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        for (int kw = 0; kw < 3; ++kw) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + ((kw * 512) + (ic_inner * 128))), 16);
          __syncthreads();
          for (int ax21 = 0; ax21 < 4; ++ax21) {
            ((int4*)(placeholder_shared + (((ax21 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + (((((((kh * 49152) + (kw * 16384)) + (ax21 * 4096)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
          }
          __syncthreads();
          for (int ax22 = 0; ax22 < 4; ++ax22) {
            (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax22], ((signed char *)placeholder_shared + (ax22 * 512)), 16);
          }
          for (int o_c = 0; o_c < 4; ++o_c) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[(((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[(((ax3_outer_inner * 32) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 8; ++ic_outer) {
    for (int ax3 = 0; ax3 < 4; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[((((((((int)blockIdx.z) * 4096) + (ic_outer * 512)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 131072) + (ax2 * 16384)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 8; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_relu)[((((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int*)placeholder2)[((((((ax0_inner_ax3_inner_fused_outer * 100352) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]), 2147483647), 0);
    }
  }
}

extern "C" __global__ void fused_cast_cast_left_shift_multiply_add_right_shift_cast_clip_cast_1_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 256) + ((int)blockIdx.x)) < 1568) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 1605632) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << (long)4) * (long)1090519040) + (long)1073741824) >> (long)31)), 127), -128));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ax3 = 0; ax3 < 4; ++ax3) {
    for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
      compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 200704) + (((int)blockIdx.z) * 64)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
    }
  }
  __syncthreads();
  for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
    __syncthreads();
    for (int ax2 = 0; ax2 < 4; ++ax2) {
      ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (ax2 * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
    }
    __syncthreads();
    for (int ax21 = 0; ax21 < 4; ++ax21) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
    }
    for (int o_c = 0; o_c < 4; ++o_c) {
      (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 802816) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = (Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__3_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 2; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) & 127) >> 4) * 65536) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 7) * 16)) + (((int)threadIdx.x) & 15)))];
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

extern "C" __global__ void fused_nn_conv2d_add_cast_add_clip_cast_nn_relu_2_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[2];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[512];
  for (int ax1_ax2_fused_inner = 0; ax1_ax2_fused_inner < 8; ++ax1_ax2_fused_inner) {
    for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
          compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[(((((((((int)blockIdx.z) * 8192) + (ax1_ax2_fused_inner * 1024)) + (ic_outer * 512)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
        }
      }
      __syncthreads();
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
        __syncthreads();
        for (int ax2 = 0; ax2 < 2; ++ax2) {
          ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 8192) + (ax2 * 4096)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
        }
        __syncthreads();
        for (int ax21 = 0; ax21 < 2; ++ax21) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
        }
        for (int o_c = 0; o_c < 2; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    for (int o_inner = 0; o_inner < 2; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
      for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
        ((int*)T_relu)[(((((((ax0_inner_ax3_inner_fused_outer * 401408) + (((int)blockIdx.z) * 4096)) + (ax1_ax2_fused_inner * 512)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int*)placeholder2)[(((((((ax0_inner_ax3_inner_fused_outer * 401408) + (((int)blockIdx.z) * 4096)) + (ax1_ax2_fused_inner * 512)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]), 2147483647), 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
    for (int ax3 = 0; ax3 < 4; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 802816) + ((((int)blockIdx.z) / 28) * 28672)) + ((((int)blockIdx.z) % 28) * 512)) + (ic_outer * 64)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 65536) + (ax2 * 8192)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 8; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((int*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 401408) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = (Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_multiply_add_right_shift_cast_nn_relu_kernel1(void* __restrict__ placeholder, signed char* __restrict__ kernel_im2col_pack, int* __restrict__ gemm_C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> gemm_C_wmma_accumulator[2];
  __shared__ signed char data_im2col_pack_shared[10240];
  __shared__ signed char kernel_im2col_pack_shared[10240];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> data_im2col_pack_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> kernel_im2col_pack_shared_wmma_matrix_b[2];
  for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
    (void)nvcuda::wmma::fill_fragment(gemm_C_wmma_accumulator[j_c_init], 0.000000e+00f);
  }
  for (int ax1_inner = 0; ax1_inner < 5; ++ax1_inner) {
    for (int ax2_ax3_fused_inner_inner_s = 0; ax2_ax3_fused_inner_inner_s < 8; ++ax2_ax3_fused_inner_inner_s) {
      data_im2col_pack_shared[((((((((int)threadIdx.y) * 2560) + (((int)threadIdx.z) * 1280)) + (ax1_inner * 256)) + (((int)threadIdx.x) * 8)) + ax2_ax3_fused_inner_inner_s))] = (((((((int)threadIdx.z) * 80) + (ax1_inner * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) & 15)) < 147) ? ((signed char*)placeholder)[((((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) >> 4)) / 12544) * 158700) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) >> 4)) % 12544) / 112) * 1380)) + (((((((int)threadIdx.z) * 80) + (ax1_inner * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) & 15)) / 21) * 690)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) >> 4)) % 112) * 6)) + ((((((int)threadIdx.z) * 80) + (ax1_inner * 16)) + (((((int)threadIdx.x) * 8) + ax2_ax3_fused_inner_inner_s) & 15)) % 21)))] : (signed char)0);
    }
  }
  for (int ax1_inner1 = 0; ax1_inner1 < 5; ++ax1_inner1) {
    ((int2*)(kernel_im2col_pack_shared + (((((((int)threadIdx.y) * 2560) + (((int)threadIdx.z) * 1280)) + (ax1_inner1 * 256)) + (((int)threadIdx.x) * 8)))))[0] = ((int2*)(kernel_im2col_pack + (((((((int)threadIdx.y) * 2560) + (((int)threadIdx.z) * 1280)) + (ax1_inner1 * 256)) + (((int)threadIdx.x) * 8)))))[0];
  }
  __syncthreads();
  for (int k1_inner = 0; k1_inner < 10; ++k1_inner) {
    (void)nvcuda::wmma::load_matrix_sync(data_im2col_pack_shared_wmma_matrix_a[0], ((signed char *)data_im2col_pack_shared + ((((int)threadIdx.y) * 2560) + (k1_inner * 256))), 16);
    for (int ax0 = 0; ax0 < 2; ++ax0) {
      (void)nvcuda::wmma::load_matrix_sync(kernel_im2col_pack_shared_wmma_matrix_b[ax0], ((signed char *)kernel_im2col_pack_shared + (((((int)threadIdx.z) * 5120) + (ax0 * 2560)) + (k1_inner * 256))), 16);
    }
    for (int j_c = 0; j_c < 2; ++j_c) {
      (void)nvcuda::wmma::mma_sync(gemm_C_wmma_accumulator[j_c], data_im2col_pack_shared_wmma_matrix_a[0], kernel_im2col_pack_shared_wmma_matrix_b[j_c], gemm_C_wmma_accumulator[j_c]);
    }
  }
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)gemm_C + ((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (j_inner * 256))), gemm_C_wmma_accumulator[j_inner], 16, nvcuda::wmma::mem_row_major);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__2_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 4; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if (((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 256) + ((int)blockIdx.x)) < 784) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 6272) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 50176) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 802816) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) & 127) >> 4) * 100352) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 7) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__6_kernel0(signed char* __restrict__ compute, void* __restrict__ placeholder) {
  for (int n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer = 0; n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer < 4; ++n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer) {
    if (((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 256) + ((int)blockIdx.x)) < 900) {
      if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 7200) {
        if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) < 57600) {
          if ((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 921600) {
            compute[((((n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[((((((((((int)threadIdx.x) & 127) >> 4) * 115200) + (n_h_fused_w_fused_i_fused_nn_fused_ii_fused_outer * 32768)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 7) * 16)) + (((int)threadIdx.x) & 15)))];
          }
        }
      }
    }
  }
}

extern "C" __global__ void fused_nn_pad_2_kernel0(void* __restrict__ T_pad, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 7200) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 921600) {
        ((signed char*)T_pad)[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((30 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) % 900)) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) % 900) < 870)) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) % 30))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) % 30) < 29)) ? ((signed char*)placeholder)[((((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) / 900) * 100352) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) % 900) / 30) * 3584)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) % 30) * 128)) + (((int)threadIdx.x) & 127)) - 3712))] : (signed char)0);
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

extern "C" __global__ void fused_nn_batch_flatten_kernel0(void* __restrict__ tensor, void* __restrict__ placeholder) {
  ((signed char*)tensor)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((signed char*)placeholder)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 16; ++ic_outer) {
    for (int ax3 = 0; ax3 < 4; ++ax3) {
      for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
        compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = ((signed char*)placeholder)[((((((((((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) >> 4) * 200704) + ((((int)blockIdx.z) / 7) * 28672)) + ((((int)blockIdx.z) % 7) * 2048)) + (ic_outer * 64)) + (ax3 * 16)) + (((((int)threadIdx.x) * 4) + ax4_ax5_fused_inner_outer) & 15)))];
      }
    }
    __syncthreads();
    for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
      (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
      __syncthreads();
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 262144) + (ax2 * 32768)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
      }
      __syncthreads();
      for (int ax21 = 0; ax21 < 8; ++ax21) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
      }
      for (int o_c = 0; o_c < 8; ++o_c) {
        (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
      }
    }
  }
  for (int o_inner = 0; o_inner < 8; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 25088) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__1_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[8];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[8];
  __shared__ int Conv[2048];
  for (int ax1_ax2_fused_inner = 0; ax1_ax2_fused_inner < 2; ++ax1_ax2_fused_inner) {
    for (int o_c_init = 0; o_c_init < 8; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    for (int ic_outer = 0; ic_outer < 4; ++ic_outer) {
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
          compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[(((((((((int)blockIdx.z) * 4096) + (ax1_ax2_fused_inner * 2048)) + (ic_outer * 512)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
        }
      }
      __syncthreads();
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
        __syncthreads();
        for (int ax2 = 0; ax2 < 8; ++ax2) {
          ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 65536) + (ax2 * 8192)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
        }
        __syncthreads();
        for (int ax21 = 0; ax21 < 8; ++ax21) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
        }
        for (int o_c = 0; o_c < 8; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    for (int o_inner = 0; o_inner < 8; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int ax3_outer_inner = 0; ax3_outer_inner < 8; ++ax3_outer_inner) {
      for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
        ((int*)T_relu)[(((((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 2048)) + (ax1_ax2_fused_inner * 1024)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 256) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int)(((((long)((int*)placeholder2)[(((((((ax0_inner_ax3_inner_fused_outer * 200704) + (((int)blockIdx.z) * 2048)) + (ax1_ax2_fused_inner * 1024)) + (((int)blockIdx.y) * 256)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_cast_cast_cast_multiply_add_right_shift_cast_add_clip_cast_n_16373524651668054328__2_kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_relu, void* __restrict__ placeholder1, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[2];
  __shared__ signed char compute_shared[512];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[1024];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  __shared__ int Conv[512];
  for (int ax1_ax2_fused_inner = 0; ax1_ax2_fused_inner < 8; ++ax1_ax2_fused_inner) {
    for (int o_c_init = 0; o_c_init < 2; ++o_c_init) {
      (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
    }
    for (int ic_outer = 0; ic_outer < 2; ++ic_outer) {
      for (int ax3 = 0; ax3 < 4; ++ax3) {
        for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
          compute_shared[((((ax3 * 128) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[(((((((((int)blockIdx.z) * 8192) + (ax1_ax2_fused_inner * 1024)) + (ic_outer * 512)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
        }
      }
      __syncthreads();
      for (int ic_inner = 0; ic_inner < 4; ++ic_inner) {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + (ic_inner * 128)), 16);
        __syncthreads();
        for (int ax2 = 0; ax2 < 2; ++ax2) {
          ((int4*)(placeholder_shared + (((ax2 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((int)blockIdx.y) * 8192) + (ax2 * 4096)) + (ic_outer * 2048)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
        }
        __syncthreads();
        for (int ax21 = 0; ax21 < 2; ++ax21) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax21], ((signed char *)placeholder_shared + (ax21 * 512)), 16);
        }
        for (int o_c = 0; o_c < 2; ++o_c) {
          (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
        }
      }
    }
    for (int o_inner = 0; o_inner < 2; ++o_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int ax3_outer_inner = 0; ax3_outer_inner < 2; ++ax3_outer_inner) {
      for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
        ((int*)T_relu)[(((((((ax0_inner_ax3_inner_fused_outer * 401408) + (((int)blockIdx.z) * 4096)) + (ax1_ax2_fused_inner * 512)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = max(min(((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 64) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) + ((int)(((((long)((int*)placeholder2)[(((((((ax0_inner_ax3_inner_fused_outer * 401408) + (((int)blockIdx.z) * 4096)) + (ax1_ax2_fused_inner * 512)) + (((int)blockIdx.y) * 64)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]) * (long)1886303204) + (long)1073741824) >> (long)31))), 2147483647), 0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_cast_cast_left_shift_multiply_add_right_shift_cast_c_3441552496575213188__kernel1(signed char* __restrict__ compute, void* __restrict__ placeholder, void* __restrict__ T_cast, void* __restrict__ placeholder1) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, int> Conv_wmma_accumulator[4];
  __shared__ signed char compute_shared[768];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, signed char, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  __shared__ signed char placeholder_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, signed char, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[4];
  __shared__ int Conv[1024];
  for (int o_c_init = 0; o_c_init < 4; ++o_c_init) {
    (void)nvcuda::wmma::fill_fragment(Conv_wmma_accumulator[o_c_init], 0.000000e+00f);
  }
  for (int ic_outer = 0; ic_outer < 16; ++ic_outer) {
    for (int kh = 0; kh < 3; ++kh) {
      for (int ax2 = 0; ax2 < 3; ++ax2) {
        for (int ax3 = 0; ax3 < 2; ++ax3) {
          for (int ax4_ax5_fused_inner_outer = 0; ax4_ax5_fused_inner_outer < 4; ++ax4_ax5_fused_inner_outer) {
            compute_shared[(((((ax2 * 256) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))] = compute[((((((((((((int)blockIdx.z) / 7) * 36864) + (kh * 36864)) + (ax2 * 4096)) + ((((int)blockIdx.z) % 7) * 4096)) + (ic_outer * 256)) + (ax3 * 128)) + (((int)threadIdx.x) * 4)) + ax4_ax5_fused_inner_outer))];
          }
        }
      }
      __syncthreads();
      for (int ic_inner = 0; ic_inner < 2; ++ic_inner) {
        for (int kw = 0; kw < 3; ++kw) {
          (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((signed char *)compute_shared + ((kw * 256) + (ic_inner * 128))), 16);
          __syncthreads();
          for (int ax21 = 0; ax21 < 4; ++ax21) {
            ((int4*)(placeholder_shared + (((ax21 * 512) + (((int)threadIdx.x) * 16)))))[0] = ((int4*)((signed char*)placeholder + ((((((((kh * 786432) + (kw * 262144)) + (((int)blockIdx.y) * 65536)) + (ax21 * 16384)) + (ic_outer * 1024)) + (ic_inner * 512)) + (((int)threadIdx.x) * 16)))))[0];
          }
          __syncthreads();
          for (int ax22 = 0; ax22 < 4; ++ax22) {
            (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax22], ((signed char *)placeholder_shared + (ax22 * 512)), 16);
          }
          for (int o_c = 0; o_c < 4; ++o_c) {
            (void)nvcuda::wmma::mma_sync(Conv_wmma_accumulator[o_c], compute_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[o_c], Conv_wmma_accumulator[o_c]);
          }
        }
      }
    }
  }
  for (int o_inner = 0; o_inner < 4; ++o_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((int *)Conv + (o_inner * 256)), Conv_wmma_accumulator[o_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int ax3_outer_inner = 0; ax3_outer_inner < 4; ++ax3_outer_inner) {
    for (int ax0_inner_ax3_inner_fused_outer = 0; ax0_inner_ax3_inner_fused_outer < 8; ++ax0_inner_ax3_inner_fused_outer) {
      ((signed char*)T_cast)[((((((ax0_inner_ax3_inner_fused_outer * 25088) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 128)) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))] = ((signed char)max(min(((int)((((((long)max((Conv[((((ax3_outer_inner * 256) + (ax0_inner_ax3_inner_fused_outer * 32)) + ((int)threadIdx.x)))] + ((int*)placeholder1)[((((((int)blockIdx.y) * 128) + (ax3_outer_inner * 32)) + ((int)threadIdx.x)))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 127), -128));
    }
  }
}

