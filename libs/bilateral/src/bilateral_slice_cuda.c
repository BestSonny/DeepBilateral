#include <TH.h>
#include <THC.h>
#include <THCGeneral.h>
#include <stdbool.h>
#include "bilateral_slice_launcher.h"

extern THCState *state;

void bilateral_slice_forward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * output){

  BilateralSliceForwardLauncher(state, grid, guide, output);

}

void bilateral_slice_backward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * grad_output,
                                  THCudaTensor * grad_grid, THCudaTensor * grad_guide){

  BilateralSliceBackwardLauncher(state, grid, guide, grad_output, grad_grid, grad_guide);

}


void bilateral_slice_apply_forward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * input,
                                       THCudaTensor * output, THCudaTensor * has_offset){

  BilateralSliceApplyForwardLauncher(state, grid, guide, input, output, has_offset);

}

void bilateral_slice_apply_backward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * input, THCudaTensor * grad_output,
                                         THCudaTensor * has_offset, THCudaTensor * grad_grid, THCudaTensor * grad_guide, THCudaTensor * grad_input){

  BilateralSliceApplyBackwardLauncher(state, grid, guide, input, grad_output, has_offset, grad_grid, grad_guide, grad_input);

}
