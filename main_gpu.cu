
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include "elas.h"
#include "GPU/elas_gpu.h"
#include "CPU/image.h"

using namespace std;
using namespace cv;
// Enable profiling
//#define PROFILE

int main(int argc, char** argv) {

  // Startup the GPU device
  // https://devtalk.nvidia.com/default/topic/895513/cuda-programming-and-performance/cudamalloc-slow/post/4724457/#4724457
  cudaFree(0);
  Mat colormap; //V the concatenated images , Right_color the is the 3 channel origianl right frame , used in viewing 
  Elas::parameters param;
 param.postprocess_only_left = false;
 ElasGPU elas(param);

  Mat leftim=imread("input/cones_left.pgm",CV_LOAD_IMAGE_GRAYSCALE);
  Mat rightim=imread("input/cones_right.pgm",CV_LOAD_IMAGE_GRAYSCALE);
  
  // get image width and height
  int32_t width  = leftim.cols;
  int32_t height = leftim.rows;
  const int32_t dims[3] = {width,height,width}; // bytes per line = width
  // allocate memory for disparity images
  float* D1_data = (float*)malloc(width*height*sizeof(float));
  float* D2_data = (float*)malloc(width*height*sizeof(float));
	cout<<"HERE\n";
  elas.process(leftim.data,rightim.data,D1_data,D2_data,dims);

  Mat L1(height, width,CV_32FC1,D1_data);
  Mat R(height, width,CV_32FC1,D2_data);
  normalize(L1, L1, 0, 255, NORM_MINMAX, CV_8U); //to view it
  applyColorMap(L1, colormap, COLORMAP_JET);  //to make it colored
	  imshow("disp",colormap);
  waitKey(0);
  free(D1_data);
  free(D2_data);


  return 0;
}