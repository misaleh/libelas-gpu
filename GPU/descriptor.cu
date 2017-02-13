/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/
/*
Edited by: Mostafa A.Saleh
moustafa.i.saleh <at> gmail.com
*/
#include "../CPU/descriptor.h"
#include "../CPU/filter.h"
#include <emmintrin.h>

#include <iostream>

using namespace std;

__global__ void sobelKernel(const uint8_t* d_in, uint8_t* d_out_v, uint8_t* d_out_h, int w, int h)
{
  int32_t x = (blockIdx.x * blockDim.x) + threadIdx.x; //width
  int32_t y = (blockIdx.y * blockDim.y) + threadIdx.y; //height

  if (x < 3 || x > w || y > h || y < 3)//out of bounds check
    return;
    const int sobel_x[3][3] = {
        {-3, 0, 3},
        {-10, 0, 10},
        {-3, 0, 3}};
    const int sobel_y[3][3]  = {
        {-3,   -10,   -3},
        {0,   0,   0},
        {3,  10,  3}};
    int16_t magnitude_x = 0 ,magnitude_y = 0 ;
    for (int16_t j = -1; j <= 1; ++j) {
        for (int16_t i = -1; i <= 1; ++i) {
            magnitude_x += d_in[(y + j)*w +(x+i)] * sobel_x[j+1][i+1];
            magnitude_y += d_in[(y + j)*w +(x+i)] * sobel_y[j+1][i+1];
        }
    }
    //magnitude_y = (magnitude_y <0)?0:magnitude_y;
    //magnitude_x = (magnitude_x <0)?0:magnitude_x;
    magnitude_y = __sad(magnitude_y,0,0); //absolute value 
    magnitude_x = __sad(magnitude_x,0,0);//absolute value 
    magnitude_y = (magnitude_y >255)?255:magnitude_y;
    magnitude_x = (magnitude_x >255)?255:magnitude_x;
    d_out_v[y*w + x ] = magnitude_y;
    d_out_h[y*w + x] = magnitude_x; 
}
void sobelGPU( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int32_t w, int32_t h )
{
    uint8_t* d_in, *d_out_h, *d_out_v;
    cudaMalloc((void**) &d_in, (w*h*sizeof(uint8_t))); //allocate input image in GPU
    cudaMalloc((void**) &d_out_h, w*h*sizeof(uint8_t)); //allocate output x image in GPU
    cudaMalloc((void**) &d_out_v, w*h*sizeof(uint8_t)); //allocate output y image in GPU
    cudaMemcpy(d_in, in, w*h*sizeof(uint8_t), cudaMemcpyHostToDevice); //copy input image to GPU
    dim3 threadsPerBlock(16,16,1);
    dim3 numBlocks( w/16, h/16,1); 
    sobelKernel<<<numBlocks , threadsPerBlock>>>(d_in,d_out_v,d_out_h,w,h);
    //cudaDeviceSynchronize();
    cudaMemcpy(out_h, d_out_h, w*h*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_v, d_out_v, w*h*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out_v);
    cudaFree(d_out_h);
    cudaFree(d_in);
}
Descriptor::Descriptor(uint8_t* I,int32_t width,int32_t height,int32_t bpl,bool half_resolution) {
  I_desc        = (uint8_t*)_mm_malloc(16*width*height*sizeof(uint8_t),16);
  uint8_t* I_du = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  uint8_t* I_dv = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  //Filter call so sobel filter to get lines better
 // filter::sobel3x3(I,I_du,I_dv,bpl,height);
   sobelGPU(I,I_du,I_dv,width,height);//fliped
  //Create 16 byte discriptors for each deep image pixel
 
  createDescriptor(I_du,I_dv,width,height,bpl,half_resolution);
  _mm_free(I_du);
  _mm_free(I_dv);
}

Descriptor::~Descriptor() {
  _mm_free(I_desc);
}

void Descriptor::createDescriptor (uint8_t* I_du,uint8_t* I_dv,int32_t width,int32_t height,int32_t bpl,bool half_resolution) {

  uint8_t *I_desc_curr;  
  uint32_t addr_v0,addr_v1,addr_v2,addr_v3,addr_v4;
  
  // do not compute every second line
  if (half_resolution) {
  
    // create filter strip
    for (int32_t v=4; v<height-3; v+=2) {

      addr_v2 = v*bpl; //Current line
      addr_v0 = addr_v2-2*bpl; //2 lines above
      addr_v1 = addr_v2-1*bpl; //1 lines above
      addr_v3 = addr_v2+1*bpl; //1 lines below
      addr_v4 = addr_v2+2*bpl; //2 lines below

      //Save the surrounding filtered rhombus point of interests (Total of 16 points)
      //Du is horizontal filter result
      //Dv is vertical filter result (more horizontal change in stero camera so we can use less vertical stuff)
      //du :
      // - - x - -
      // - x x x -
      // x x o x x
      // - x x x -
      // - - x - -
      //dv :
      // - - - - -
      // - - x - -
      // - x o x -
      // - - x - -
      // - - - - -
      for (int32_t u=3; u<width-3; u++) {
        I_desc_curr = I_desc+(v*width+u)*16;
        *(I_desc_curr++) = *(I_du+addr_v0+u+0);
        *(I_desc_curr++) = *(I_du+addr_v1+u-2);
        *(I_desc_curr++) = *(I_du+addr_v1+u+0);
        *(I_desc_curr++) = *(I_du+addr_v1+u+2);
        *(I_desc_curr++) = *(I_du+addr_v2+u-1);
        *(I_desc_curr++) = *(I_du+addr_v2+u+0);
        *(I_desc_curr++) = *(I_du+addr_v2+u+0);
        *(I_desc_curr++) = *(I_du+addr_v2+u+1);
        *(I_desc_curr++) = *(I_du+addr_v3+u-2);
        *(I_desc_curr++) = *(I_du+addr_v3+u+0);
        *(I_desc_curr++) = *(I_du+addr_v3+u+2);
        *(I_desc_curr++) = *(I_du+addr_v4+u+0);
        *(I_desc_curr++) = *(I_dv+addr_v1+u+0);
        *(I_desc_curr++) = *(I_dv+addr_v2+u-1);
        *(I_desc_curr++) = *(I_dv+addr_v2+u+1);
        *(I_desc_curr++) = *(I_dv+addr_v3+u+0);
      }
    }
    
  // compute full descriptor images
  } else {
    
    // create filter strip
    for (int32_t v=3; v<height-3; v++) {

      addr_v2 = v*bpl;
      addr_v0 = addr_v2-2*bpl;
      addr_v1 = addr_v2-1*bpl;
      addr_v3 = addr_v2+1*bpl;
      addr_v4 = addr_v2+2*bpl;

      for (int32_t u=3; u<width-3; u++) {
        I_desc_curr = I_desc+(v*width+u)*16;
        *(I_desc_curr++) = *(I_du+addr_v0+u+0);
        *(I_desc_curr++) = *(I_du+addr_v1+u-2);
        *(I_desc_curr++) = *(I_du+addr_v1+u+0);
        *(I_desc_curr++) = *(I_du+addr_v1+u+2);
        *(I_desc_curr++) = *(I_du+addr_v2+u-1);
        *(I_desc_curr++) = *(I_du+addr_v2+u+0);
        *(I_desc_curr++) = *(I_du+addr_v2+u+0);
        *(I_desc_curr++) = *(I_du+addr_v2+u+1);
        *(I_desc_curr++) = *(I_du+addr_v3+u-2);
        *(I_desc_curr++) = *(I_du+addr_v3+u+0);
        *(I_desc_curr++) = *(I_du+addr_v3+u+2);
        *(I_desc_curr++) = *(I_du+addr_v4+u+0);
        *(I_desc_curr++) = *(I_dv+addr_v1+u+0);
        *(I_desc_curr++) = *(I_dv+addr_v2+u-1);
        *(I_desc_curr++) = *(I_dv+addr_v2+u+1);
        *(I_desc_curr++) = *(I_dv+addr_v3+u+0);
      }
    }
  }
  
}

