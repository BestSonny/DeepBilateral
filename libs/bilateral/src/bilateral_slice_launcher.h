#ifndef _BILATERAL_SLICE_KERNEL
#define _BILATERAL_SLICE_KERNEL

#include <THC.h>
#include <THCGeneral.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void BilateralSliceForwardKernel(const int nthreads, const int batch_size, const int height, const int width,
                                  const int depth, const int nchannels, const int guide_height, const int guide_width,
                                  const float* grid, const float* guide, float* output);

void BilateralSliceForwardLauncher(THCState* state, THCudaTensor* grid, THCudaTensor* guide, THCudaTensor* output);

__global__ void BilateralSliceBackwardKernel(const int nthreads, const int batch_size, const int height, const int width,
                                  const int depth, const int nchannels, const int guide_height, const int guide_width,
                                  const float* grid, const float* guide, const float* grad_output,
                                  float* grad_grid, float* grad_guide);


void BilateralSliceBackwardLauncher(THCState* state, THCudaTensor* grid, THCudaTensor* guide, THCudaTensor* grad_output,
                                    THCudaTensor* grad_grid, THCudaTensor* grad_guide);


__global__ void BilateralSliceApplyForwardKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                   const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                   const float* grid, const float* guide, const float* input, float* output);

void BilateralSliceApplyForwardLauncher(THCState* state, THCudaTensor* grid, THCudaTensor* guide, THCudaTensor* input, THCudaTensor* output, THCudaTensor* has_offset);


__global__ void BilateralSliceApplyBackwardGridKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                   const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                  const float* grid, const float* guide, const float* input, const float* grad_output,
                                  float* grad_grid);

__global__ void BilateralSliceApplyBackwardGuideKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                   const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                  const float* grid, const float* guide, const float* input, const float* grad_output,
                                  float* grad_guide);

__global__ void BilateralSliceApplyBackwardInputKernel(const int nthreads, const int batch_size, const int height, const int width, const int depth,
                                   const int input_chans, const int output_chans, const int guide_height, const int guide_width, const int has_offset,
                                   const float* grid, const float* guide, const float* input, const float* grad_output,
                                   float* grad_input);

void BilateralSliceApplyBackwardLauncher(THCState* state, THCudaTensor* grid, THCudaTensor* guide, THCudaTensor* input, THCudaTensor* grad_output,
                                         THCudaTensor* has_offset, THCudaTensor* grad_grid, THCudaTensor* grad_guide, THCudaTensor* grad_input);

#ifdef __cplusplus
}
#endif

#endif
