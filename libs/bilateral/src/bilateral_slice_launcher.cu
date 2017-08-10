#include <THC.h>
#include <THCGeneral.h>
#include "bilateral_slice_launcher.h"
#ifdef __cplusplus
extern "C" {
#endif


#define CUDA_KERNEL_LOOP(index, nthreads)                            \
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; \
            index += blockDim.x * gridDim.x)

__global__ void BilateralSliceForwardKernel(const int nthreads, const int batch_size, const int height, const int width,
                                  const int depth, const int nchannels, const int guide_height, const int guide_width,
                                  const float* grid, const float* guide, float* output){
  CUDA_KERNEL_LOOP(index, nthreads){
    int c = index % nchannels;
    int x = (index / nchannels) % guide_width;
    int y = (index / (nchannels * guide_width)) % guide_height;
    int b = (index / (nchannels * guide_width * guide_height));

    const float rheight = (guide_height > 1) ? (float)(height - 1.0f)/(guide_height - 1.0f) : 0.0f;
    const float rwidth = (guide_width > 1) ? (float)(width - 1.0f)/(guide_width - 1.0f) : 0.0f;

    float gx = x * rwidth;
    const int w1 = gx;
    const int w1p = (w1 < width - 1) ? 1 : 0;
    const float w1lambda = gx - w1; // right
    const float w0lambda = 1.0f - w1lambda; // left

    float gy = y * rheight;
    const int h1 = gy;
    const int h1p = (h1 < height - 1) ? 1 : 0;
    const float h1lambda = gy - h1; // bottom
    const float h0lambda = 1.0f - h1lambda; //top


    float gz = guide[x + guide_width * (y + guide_height * b)] * depth;
    const int z1 = gz;
    int z1p = (z1 < depth - 1) ? 1 : 0;
    float z1lambda = gz - z1; // higher
    float z0lambda = 1-z1lambda; // lower

    int sz = nchannels;
    int sx = nchannels*depth;
    int sy = nchannels*depth*width;
    int sb = nchannels*depth*width*height;

    const float value = z0lambda * h0lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b] +
                        z0lambda * h0lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b] +
                        z0lambda * h1lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b] +
                        z0lambda * h1lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b] +
                        z1lambda * h0lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b] +
                        z1lambda * h0lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b] +
                        z1lambda * h1lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b] +
                        z1lambda * h1lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];

    output[index] = value;
  }
}


void BilateralSliceForwardLauncher(THCState* state, THCudaTensor* grid, THCudaTensor* guide, THCudaTensor* output){

  const int kThreadsPerBlock = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  const int nthreads = THCudaTensor_nElement(state, output);
  const int batch_size = THCudaTensor_size(state, grid, 0);
  const int height = THCudaTensor_size(state, grid, 1);
  const int width = THCudaTensor_size(state, grid, 2);
  const int depth = THCudaTensor_size(state, grid, 3);
  const int nchannels = THCudaTensor_size(state, grid, 4);
  const int guide_height = THCudaTensor_size(state, guide, 1);
  const int guide_width = THCudaTensor_size(state, guide, 2);

  BilateralSliceForwardKernel<<<(nthreads + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, THCState_getCurrentStream(state)>>>(nthreads, batch_size, height, width, depth,
                             nchannels, guide_height, guide_width,
                             THCudaTensor_data(state, grid),
                             THCudaTensor_data(state, guide),
                             THCudaTensor_data(state, output));

  cudaDeviceSynchronize();
  THCudaCheck(cudaGetLastError());
}

__global__ void BilateralSliceBackwardKernel(const int nthreads, const int batch_size, const int height, const int width,
                                  const int depth, const int nchannels, const int guide_height, const int guide_width,
                                  const float* grid, const float* guide, const float* grad_output,
                                  float* grad_grid, float* grad_guide){
 CUDA_KERNEL_LOOP(index, nthreads) {
   int c = index % nchannels;
   int x = (index / nchannels) % guide_width;
   int y = (index / (nchannels * guide_width)) % guide_height;
   int b = (index / (nchannels * guide_width * guide_height));

   const float rheight = (guide_height > 1) ? (float)(height - 1.0f)/(guide_height - 1.0f) : 0.0f;
   const float rwidth = (guide_width > 1) ? (float)(width - 1.0f)/(guide_width - 1.0f) : 0.0f;

   float gx = x * rwidth;
   const int w1 = gx;
   const int w1p = (w1 < width - 1) ? 1 : 0;
   const float w1lambda = gx - w1; // right
   const float w0lambda = 1.0f - w1lambda; // left

   float gy = y * rheight;
   const int h1 = gy;
   const int h1p = (h1 < height - 1) ? 1 : 0;
   const float h1lambda = gy - h1; // bottom
   const float h0lambda = 1.0f - h1lambda; //top

   float gz = guide[x + guide_width * (y + guide_height * b)] * depth;
   const int z1 = gz;
   int z1p = (z1 < depth - 1) ? 1 : 0;
   float z1lambda = gz - z1; // higher
   float z0lambda = 1 - z1lambda; // lower

   int sz = nchannels;
   int sx = nchannels*depth;
   int sy = nchannels*depth*width;
   int sb = nchannels*depth*width*height;

   const float value_z0 = h0lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b] +
                          h0lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b] +
                          h1lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b] +
                          h1lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];

   const float value_z1 = h0lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b] +
                          h0lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b] +
                          h1lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b] +
                          h1lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];

   // compute grad_guide
   atomicAdd(&grad_guide[x + guide_width * (y + guide_height * b)], depth * (value_z1 - value_z0) * grad_output[index]);

   // compute grad_grid
   atomicAdd(&grad_grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b],
             z0lambda * h0lambda * w0lambda * grad_output[index]);
   atomicAdd(&grad_grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b],
             z0lambda * h0lambda * w1lambda * grad_output[index]);
   atomicAdd(&grad_grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b],
             z0lambda * h1lambda * w0lambda * grad_output[index]);
   atomicAdd(&grad_grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b],
             z0lambda * h1lambda * w1lambda * grad_output[index]);

   atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b],
             z1lambda * h0lambda * w0lambda * grad_output[index]);
   atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b],
             z1lambda * h0lambda * w1lambda * grad_output[index]);
   atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b],
             z1lambda * h1lambda * w0lambda * grad_output[index]);
   atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b],
             z1lambda * h1lambda * w1lambda * grad_output[index]);

  }
}

void BilateralSliceBackwardLauncher(THCState* state, THCudaTensor* grid, THCudaTensor* guide, THCudaTensor* grad_output,
                                  THCudaTensor* grad_grid, THCudaTensor* grad_guide){

  const int kThreadsPerBlock = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  const int batch_size = THCudaTensor_size(state, grid, 0);
  const int height = THCudaTensor_size(state, grid, 1);
  const int width = THCudaTensor_size(state, grid, 2);
  const int depth = THCudaTensor_size(state, grid, 3);
  const int nchannels = THCudaTensor_size(state, grid, 4);
  const int guide_height = THCudaTensor_size(state, guide, 1);
  const int guide_width = THCudaTensor_size(state, guide, 2);
  const int nthreads = THCudaTensor_nElement(state, grad_output);

  BilateralSliceBackwardKernel<<<(nthreads + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, THCState_getCurrentStream(state)>>>(nthreads, batch_size, height, width, depth,
                             nchannels, guide_height, guide_width,
                             THCudaTensor_data(state, grid), THCudaTensor_data(state, guide),
                             THCudaTensor_data(state, grad_output), THCudaTensor_data(state, grad_grid), THCudaTensor_data(state, grad_guide));

  cudaDeviceSynchronize();
  THCudaCheck(cudaGetLastError());
}



__global__ void BilateralSliceApplyForwardKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                   const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                   const float* grid, const float* guide, const float* input, float* output){

   int grid_chans = input_chans*output_chans;
   int coeff_stride = input_chans;

   if(has_offset>0) {
     grid_chans += output_chans;
     coeff_stride += 1;
   }

  CUDA_KERNEL_LOOP(index, nthreads) {
    int out_c = index % output_chans;
    int x = (index / output_chans) % guide_width;
    int y = (index / (output_chans * guide_width)) % guide_height;
    int b = (index / (output_chans * guide_width * guide_height));

    const float rheight = (guide_height > 1) ? (float)(height - 1.0f)/(guide_height - 1.0f) : 0.0f;
    const float rwidth = (guide_width > 1) ? (float)(width - 1.0f)/(guide_width - 1.0f) : 0.0f;

    float gx = x * rwidth;
    const int w1 = gx;
    const int w1p = (w1 < width - 1) ? 1 : 0;
    const float w1lambda = gx - w1; // right
    const float w0lambda = 1.0f - w1lambda; // left

    float gy = y * rheight;
    const int h1 = gy;
    const int h1p = (h1 < height - 1) ? 1 : 0;
    const float h1lambda = gy - h1; // bottom
    const float h0lambda = 1.0f - h1lambda; //top


    float gz = guide[x + guide_width * (y + guide_height * b)] * depth;
    const int z1 = gz;
    int z1p = (z1 < depth - 1) ? 1 : 0;
    float z1lambda = gz - z1; // higher
    float z0lambda = 1-z1lambda; // lower

    int sz = grid_chans;
    int sx = grid_chans*depth;
    int sy = grid_chans*depth*width;
    int sb = grid_chans*depth*width*height;


    float value = 0;
    int in_c = 0;

    for (; in_c < coeff_stride; ++in_c) {
      int c = coeff_stride * out_c + in_c;
      float coeff_sample = z0lambda * h0lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b] +
                           z0lambda * h0lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b] +
                           z0lambda * h1lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b] +
                           z0lambda * h1lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b] +
                           z1lambda * h0lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b] +
                           z1lambda * h0lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b] +
                           z1lambda * h1lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b] +
                           z1lambda * h1lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];
      if(in_c < input_chans) {
        int input_idx = in_c + input_chans*(x + guide_width*(y + guide_height*b));
        value += coeff_sample*input[input_idx];
      } else { // Offset term
        value += coeff_sample;
      }
    }
    output[index] = value;

  }
}

void BilateralSliceApplyForwardLauncher(THCState* state, THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * input,
                                       THCudaTensor * output, THCudaTensor * has_offset){

  const int flag = THCudaTensor_size(state, has_offset, 0);
  const int batch_size = THCudaTensor_size(state, grid, 0);
  const int height = THCudaTensor_size(state, grid, 1);
  const int width = THCudaTensor_size(state, grid, 2);
  const int depth = THCudaTensor_size(state, grid, 3);
  const int nchannels = THCudaTensor_size(state, grid, 4);
  const int guide_height = THCudaTensor_size(state, guide, 1);
  const int guide_width = THCudaTensor_size(state, guide, 2);
  const int input_channels = THCudaTensor_size(state, input, 3);
  int output_channel = 0;
  if(flag == 1){
    THAssert(nchannels % (input_channels + 1) == 0);
    output_channel = nchannels / (input_channels + 1);
  }else{
    THAssert(nchannels % input_channels == 0);
    output_channel = nchannels / input_channels;
  }

  THCudaTensor_resize4d(state, output, batch_size, guide_height, guide_width, output_channel);
  THCudaTensor_zero(state, output);

  const int kThreadsPerBlock = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  const int nthreads = THCudaTensor_nElement(state, output);


  BilateralSliceApplyForwardKernel<<<(nthreads + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, THCState_getCurrentStream(state)>>>(nthreads, batch_size, height, width, depth,
                                   input_channels, output_channel, guide_height, guide_width, flag,
                                   THCudaTensor_data(state, grid), THCudaTensor_data(state, guide),
                                   THCudaTensor_data(state, input), THCudaTensor_data(state, output));

  cudaDeviceSynchronize();
  THCudaCheck(cudaGetLastError());

}


__global__ void BilateralSliceApplyBackwardGridKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                  const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                  const float* grid, const float* guide, const float* input, const float* grad_output,
                                  float* grad_grid){
  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;

  if(has_offset>0) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

 CUDA_KERNEL_LOOP(index, nthreads) {
   int out_c = index % output_chans;
   int x = (index / output_chans) % guide_width;
   int y = (index / (output_chans * guide_width)) % guide_height;
   int b = (index / (output_chans * guide_width * guide_height));

   const float rheight = (guide_height > 1) ? (float)(height - 1.0f)/(guide_height - 1.0f) : 0.0f;
   const float rwidth = (guide_width > 1) ? (float)(width - 1.0f)/(guide_width - 1.0f) : 0.0f;

   float gx = x * rwidth;
   const int w1 = gx;
   const int w1p = (w1 < width - 1) ? 1 : 0;
   const float w1lambda = gx - w1; // right
   const float w0lambda = 1.0f - w1lambda; // left

   float gy = y * rheight;
   const int h1 = gy;
   const int h1p = (h1 < height - 1) ? 1 : 0;
   const float h1lambda = gy - h1; // bottom
   const float h0lambda = 1.0f - h1lambda; //top


   float gz = guide[x + guide_width * (y + guide_height * b)] * depth;
   const int z1 = gz;
   int z1p = (z1 < depth - 1) ? 1 : 0;
   float z1lambda = gz - z1; // higher
   float z0lambda = 1-z1lambda; // lower

   int sz = grid_chans;
   int sx = grid_chans*depth;
   int sy = grid_chans*depth*width;
   int sb = grid_chans*depth*width*height;

   int in_c = 0;

   for (; in_c < coeff_stride; ++in_c) {
      int c = coeff_stride * out_c + in_c;

      if(in_c < input_chans) {
       int input_idx = in_c + input_chans*(x + guide_width*(y + guide_height*b));
       // compute grad_grid
       atomicAdd(&grad_grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b],
                 z0lambda * h0lambda * w0lambda * grad_output[index] * input[input_idx]);
       atomicAdd(&grad_grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b],
                 z0lambda * h0lambda * w1lambda * grad_output[index] * input[input_idx]);
       atomicAdd(&grad_grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b],
                 z0lambda * h1lambda * w0lambda * grad_output[index] * input[input_idx]);
       atomicAdd(&grad_grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b],
                 z0lambda * h1lambda * w1lambda * grad_output[index] * input[input_idx]);

       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b],
                 z1lambda * h0lambda * w0lambda * grad_output[index] * input[input_idx]);
       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b],
                 z1lambda * h0lambda * w1lambda * grad_output[index] * input[input_idx]);
       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b],
                 z1lambda * h1lambda * w0lambda * grad_output[index] * input[input_idx]);
       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b],
                 z1lambda * h1lambda * w1lambda * grad_output[index] * input[input_idx]);

      } else { // Offset term

       atomicAdd(&grad_grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b],
                 z0lambda * h0lambda * w0lambda * grad_output[index]);
       atomicAdd(&grad_grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b],
                 z0lambda * h0lambda * w1lambda * grad_output[index]);
       atomicAdd(&grad_grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b],
                 z0lambda * h1lambda * w0lambda * grad_output[index]);
       atomicAdd(&grad_grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b],
                 z0lambda * h1lambda * w1lambda * grad_output[index]);

       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b],
                 z1lambda * h0lambda * w0lambda * grad_output[index]);
       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b],
                 z1lambda * h0lambda * w1lambda * grad_output[index]);
       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b],
                 z1lambda * h1lambda * w0lambda * grad_output[index]);
       atomicAdd(&grad_grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b],
                 z1lambda * h1lambda * w1lambda * grad_output[index]);
      }
    }
  }
}

__global__ void BilateralSliceApplyBackwardGuideKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                  const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                  const float* grid, const float* guide, const float* input, const float* grad_output,
                                  float* grad_guide){
  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;

  if(has_offset>0) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

 CUDA_KERNEL_LOOP(index, nthreads) {
   int out_c = index % output_chans;
   int x = (index / output_chans) % guide_width;
   int y = (index / (output_chans * guide_width)) % guide_height;
   int b = (index / (output_chans * guide_width * guide_height));

   const float rheight = (guide_height > 1) ? (float)(height - 1.0f)/(guide_height - 1.0f) : 0.0f;
   const float rwidth = (guide_width > 1) ? (float)(width - 1.0f)/(guide_width - 1.0f) : 0.0f;

   float gx = x * rwidth;
   const int w1 = gx;
   const int w1p = (w1 < width - 1) ? 1 : 0;
   const float w1lambda = gx - w1; // right
   const float w0lambda = 1.0f - w1lambda; // left

   float gy = y * rheight;
   const int h1 = gy;
   const int h1p = (h1 < height - 1) ? 1 : 0;
   const float h1lambda = gy - h1; // bottom
   const float h0lambda = 1.0f - h1lambda; //top


   float gz = guide[x + guide_width * (y + guide_height * b)] * depth;
   const int z1 = gz;
   int z1p = (z1 < depth - 1) ? 1 : 0;


   int sz = grid_chans;
   int sx = grid_chans*depth;
   int sy = grid_chans*depth*width;
   int sb = grid_chans*depth*width*height;

   int in_c = 0;

   for (; in_c < coeff_stride; ++in_c) {
      int c = coeff_stride * out_c + in_c;

      const float value_z0 = h0lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b] +
                             h0lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b] +
                             h1lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b] +
                             h1lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];

      const float value_z1 = h0lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b] +
                             h0lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b] +
                             h1lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b] +
                             h1lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];

      if(in_c < input_chans) {
       int input_idx = in_c + input_chans*(x + guide_width*(y + guide_height*b));
       // compute grad_guide
       atomicAdd(&grad_guide[x + guide_width * (y + guide_height * b)], depth * (value_z1 - value_z0) * grad_output[index] * input[input_idx]);

      } else { // Offset term

       atomicAdd(&grad_guide[x + guide_width * (y + guide_height * b)], depth * (value_z1 - value_z0) * grad_output[index]);

      }
    }
  }
}

__global__ void BilateralSliceApplyBackwardInputKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                  const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                  const float* grid, const float* guide, const float* input, const float* grad_output,
                                  float* grad_input){
  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;

  if(has_offset>0) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

 CUDA_KERNEL_LOOP(index, nthreads) {
   int out_c = index % output_chans;
   int x = (index / output_chans) % guide_width;
   int y = (index / (output_chans * guide_width)) % guide_height;
   int b = (index / (output_chans * guide_width * guide_height));

   const float rheight = (guide_height > 1) ? (float)(height - 1.0f)/(guide_height - 1.0f) : 0.0f;
   const float rwidth = (guide_width > 1) ? (float)(width - 1.0f)/(guide_width - 1.0f) : 0.0f;

   float gx = x * rwidth;
   const int w1 = gx;
   const int w1p = (w1 < width - 1) ? 1 : 0;
   const float w1lambda = gx - w1; // right
   const float w0lambda = 1.0f - w1lambda; // left

   float gy = y * rheight;
   const int h1 = gy;
   const int h1p = (h1 < height - 1) ? 1 : 0;
   const float h1lambda = gy - h1; // bottom
   const float h0lambda = 1.0f - h1lambda; //top


   float gz = guide[x + guide_width * (y + guide_height * b)] * depth;
   const int z1 = gz;
   int z1p = (z1 < depth - 1) ? 1 : 0;
   float z1lambda = gz - z1; // higher
   float z0lambda = 1-z1lambda; // lower

   int sz = grid_chans;
   int sx = grid_chans*depth;
   int sy = grid_chans*depth*width;
   int sb = grid_chans*depth*width*height;

   int in_c = 0;

   for (; in_c < coeff_stride; ++in_c) {
      int c = coeff_stride * out_c + in_c;

      const float value_z0 = h0lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*h1 + sb*b] +
                             h0lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*h1 + sb*b] +
                             h1lambda * w0lambda * grid[c + sz*z1 + sx*w1 + sy*(h1+h1p) + sb*b] +
                             h1lambda * w1lambda * grid[c + sz*z1 + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];

      const float value_z1 = h0lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*h1 + sb*b] +
                             h0lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*h1 + sb*b] +
                             h1lambda * w0lambda * grid[c + sz*(z1+z1p) + sx*w1 + sy*(h1+h1p) + sb*b] +
                             h1lambda * w1lambda * grid[c + sz*(z1+z1p) + sx*(w1+w1p) + sy*(h1+h1p) + sb*b];

      if(in_c < input_chans) {
       int input_idx = in_c + input_chans*(x + guide_width*(y + guide_height*b));
       // compute grad_input
       float coeff_sample = z0lambda * value_z0 + z1lambda * value_z1;
       atomicAdd(&grad_input[input_idx], coeff_sample * grad_output[index]);

      }
    }
  }
}

void BilateralSliceApplyBackwardLauncher(THCState* state, THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * input, THCudaTensor * grad_output,
                                         THCudaTensor* has_offset, THCudaTensor * grad_grid, THCudaTensor * grad_guide, THCudaTensor * grad_input){
   const int flag = THCudaTensor_size(state, has_offset, 0);
   const int batch_size = THCudaTensor_size(state, grid, 0);
   const int height = THCudaTensor_size(state, grid, 1);
   const int width = THCudaTensor_size(state, grid, 2);
   const int depth = THCudaTensor_size(state, grid, 3);
   const int nchannels = THCudaTensor_size(state, grid, 4);
   const int guide_height = THCudaTensor_size(state, guide, 1);
   const int guide_width = THCudaTensor_size(state, guide, 2);
   const int input_channels = THCudaTensor_size(state, input, 3);
   int output_channel = 0;
   if(flag == 1){
     THAssert(nchannels % (input_channels + 1) == 0);
     output_channel = nchannels / (input_channels + 1);
   }else{
     THAssert(nchannels % input_channels == 0);
     output_channel = nchannels / input_channels;
   }
   const int kThreadsPerBlock = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
   const int nthreads = THCudaTensor_nElement(state, grad_output);

   BilateralSliceApplyBackwardGridKernel<<<(nthreads + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, THCState_getCurrentStream(state)>>>(nthreads, batch_size, height, width, depth,
                                    input_channels, output_channel, guide_height, guide_width, flag,
                                    THCudaTensor_data(state, grid), THCudaTensor_data(state, guide),
                                    THCudaTensor_data(state, input), THCudaTensor_data(state, grad_output),
                                    THCudaTensor_data(state, grad_grid));
   BilateralSliceApplyBackwardGuideKernel<<<(nthreads + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, THCState_getCurrentStream(state)>>>(nthreads, batch_size, height, width, depth,
                                    input_channels, output_channel, guide_height, guide_width, flag,
                                    THCudaTensor_data(state, grid), THCudaTensor_data(state, guide),
                                    THCudaTensor_data(state, input), THCudaTensor_data(state, grad_output),
                                    THCudaTensor_data(state, grad_guide));
   BilateralSliceApplyBackwardInputKernel<<<(nthreads + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, THCState_getCurrentStream(state)>>>(nthreads, batch_size, height, width, depth,
                                    input_channels, output_channel, guide_height, guide_width, flag,
                                    THCudaTensor_data(state, grid), THCudaTensor_data(state, guide),
                                    THCudaTensor_data(state, input), THCudaTensor_data(state, grad_output),
                                    THCudaTensor_data(state, grad_input));
   cudaDeviceSynchronize();
   THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
}
#endif
