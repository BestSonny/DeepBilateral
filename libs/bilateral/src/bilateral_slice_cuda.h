void bilateral_slice_forward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * output);

void bilateral_slice_backward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * grad_output,
                                   THCudaTensor * grad_grid, THCudaTensor * grad_guide);

void bilateral_slice_apply_forward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * input,
                                       THCudaTensor * output, THCudaTensor * has_offset);

void bilateral_slice_apply_backward_cuda(THCudaTensor * grid, THCudaTensor * guide, THCudaTensor * input, THCudaTensor * grad_output,
                                         THCudaTensor * has_offset, THCudaTensor * grad_grid, THCudaTensor * grad_guide, THCudaTensor * grad_input);
