#include "luaT.h"
#include "THC.h"

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

//#include "SpatialSoftMax2.cu"

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit


struct expupdateOutput_functor
{
    __host__ __device__ float operator()(const float& input) const
    {
    return exp(input);
    }
};

//no-overlap 
__global__ void softmax(float *input, float *output,
                        int input_n, int input_h, int input_w,
                        int kH, int kW)
{
    //select the block
    float* ptr_input_plane = input + blockIdx.x * input_w * input_h;
    float* ptr_output_plane = output + blockIdx.x * input_w * input_h;
    
    //iterate inside the block
    int x_start = threadIdx.x * kW;
    int y_start = threadIdx.y * kH; 
    int x_step = blockDim.x * kW;
    int y_step = blockDim.y * kH;

    for (int y = y_start; y < input_h; y += y_step) {
        for (int x = x_start; x < input_w; x += x_step) {
        float* ptr_input = ptr_input_plane + x + y * input_w;
        float* ptr_output = ptr_output_plane + x + y * input_w;
        float pool_sum = 0; 

          for (int ky = 0; ky < kH && y + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && x + kx < input_w; kx++) {
              float* ptr_input_pool = ptr_input + kx + ky * input_w;
              pool_sum += exp(*ptr_input_pool);  
            }
          }
          
          for (int ky = 0; ky < kH && y + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && x + kx < input_w; kx++) {
              float* ptr_input_pool = ptr_input + kx + ky * input_w;
              float* ptr_output_pool = ptr_output + kx + ky * input_w;
              *ptr_output_pool = exp(*ptr_input_pool)/pool_sum;  
            }
          }
       
        }
    }

}

__global__ void softmaxgradinput(float *gradInput, float *gradOutput, float *output,
                             int input_n, int input_h, int input_w,
                             int kH, int kW)
{

    //select the block
    float* ptr_gradInput_plane = gradInput + blockIdx.x * input_w * input_h;
    float* ptr_gradOutput_plane = gradOutput + blockIdx.x * input_w * input_h;
    float* ptr_output_plane = output + blockIdx.x * input_w * input_h;
    
    //iterate inside the block
    int x_start = threadIdx.x * kW;
    int y_start = threadIdx.y * kH; 
    int x_step = blockDim.x * kW;
    int y_step = blockDim.y * kH;

    for (int y = y_start; y < input_h; y += y_step) {
        for (int x = x_start; x < input_w; x += x_step) {
        float* ptr_gradInput = ptr_gradInput_plane + x + y * input_w;
        float* ptr_gradOutput = ptr_gradOutput_plane + x + y * input_w;
        float* ptr_output = ptr_output_plane + x + y * input_w;
        float pool_sum = 0; 

          for (int ky = 0; ky < kH && y + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && x + kx < input_w; kx++) {
              float* ptr_gradOutput_pool = ptr_gradOutput + kx + ky * input_w;
              float* ptr_output_pool = ptr_output + kx + ky * input_w;
              pool_sum += (*ptr_output_pool) * (*ptr_gradOutput_pool);  
            }
          }
       
          
          for (int ky = 0; ky < kH && y + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && x + kx < input_w; kx++) {
              float* ptr_gradOutput_pool = ptr_gradOutput + kx + ky * input_w;
              float* ptr_gradInput_pool = ptr_gradInput + kx + ky * input_w;
              float* ptr_output_pool = ptr_output + kx + ky * input_w;
              *ptr_gradInput_pool = (*ptr_output_pool) * ((*ptr_gradOutput_pool) - pool_sum);  
            }
          }
        
        
        }
    }
}

static int cunn_SpatialSoftMax_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");

  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  float *output_data;
  float *input_data;

  long nInputCols = input->size[3];
  long nInputRows = input->size[2];
  long nInputPlane = input->size[1];
  long nbatch = input->size[0];

  luaL_argcheck(L, input->size[1] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  input = THCudaTensor_newContiguous(input);
  input_data = THCudaTensor_data(input);

  THCudaTensor_resize4d(output, nbatch, nInputPlane, nInputRows, nInputCols);

  output_data = THCudaTensor_data(output);

  // cuda blocks & threads:
  dim3 blocks(nInputPlane*nbatch,1);
  dim3 threads(32,8);

  // run maxpool kernel
  softmax <<<blocks, threads>>> (input_data, output_data,
                                   nInputPlane, nInputRows, nInputCols, kH, kW);
  // clean
  THCudaTensor_free(input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSoftMax.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_SpatialSoftMax_updateGradInput(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");

  float *gradInput_data;
  float *gradOutput_data;
  float *output_data;

  long nInputCols = input->size[3];
  long nInputRows = input->size[2];
  long nInputPlane = input->size[1];
  long nbatch = input->size[0];
  long nOutputCols = gradOutput->size[3];
  long nOutputRows = gradOutput->size[2];

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_zero(gradInput);

  gradOutput_data = THCudaTensor_data(gradOutput);
  gradInput_data = THCudaTensor_data(gradInput);
  output_data = THCudaTensor_data(output);

  dim3 blocks(nInputPlane*nbatch,1);
  dim3 threads(32,8);

  // run updateGradInput kernel
  softmaxgradinput<<<blocks, threads>>>
	(gradInput_data, gradOutput_data, output_data,
	 nInputPlane, nInputRows, nInputCols, kH, kW);
  

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSoftMaxsampling.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static const struct luaL_Reg cunn_SpatialSoftMax__ [] = {
  {"SpatialSoftMax_updateOutput", cunn_SpatialSoftMax_updateOutput},
  {"SpatialSoftMax_updateGradInput", cunn_SpatialSoftMax_updateGradInput},
  {NULL, NULL}
};

void cunn_SpatialSoftMax_init(lua_State *L)
{
  luaL_openlib(L, "jzt", cunn_SpatialSoftMax__, 0);
}
