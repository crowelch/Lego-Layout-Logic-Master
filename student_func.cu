//Lego Layout Logic Master
//Cuda Functions

#include "reference_calc.cpp"
#include "utils.h"

__global__
void averageLegoBlockColor(const uchar4* const rgbaImage,
                           unsigned char* const greyImage,
                           int numRows, 
                           int numCols, 
                           const int legoSize)
{
  //Array of lego colors.  In order:
  //Red, Green, Blue, Cyan, Magenta, Yellow, White. Gray, Black	
  int legoColorArray[20] = [255,65280,16711680,16776960,16711935,
  	                        65535,16777215,8421504,0]
  __shared__ int legoBlockNums[10000000];
  //__shared__ int legoBlockNums[numRows * numCols];
  int absolute_image_position_x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int absolute_image_position_y = (blockDim.y * blockIdx.y) + threadIdx.y; 
  int id = absolute_image_position_x + (absolute_image_position_y * numCols);
  int legoRowNum = (absolute_image_position_x / legoSize) + 1;
  int legoColumnNum = (absolute_image_position_y / legoSize) + 1;
  int legoPieceId = legoRowNum + (legoColumnNum * (numCols/legoSize));
  int blockNums[10000000];
  blockNums[id] = legoPieceId;
  __syncthreads();
  float redAverage = 0;
  float greenAverage = 0;
  float blueAverage = 0;
  int numPixelsAveraged;
  for(int i = 0; i < numRows*numCols; i++) //loop should find the pixel itself
  {
    if(legoBlockNums[i] == legoPieceId)
		{
		  redAverage = redAverage + rgbaImage[i].x;
			greenAverage = greenAverage + rgbaImage[i].y;
		  blueAverage = blueAverage + rgbaImage[i].z;
			numPixelsAveraged ++;
		}	
  }
  //convert rgb colors to a single integer color
  int rgb_to_int = redAverage;
  rgb_to_int = (rgb_to_int << 8) + greenAverage;
  rgb_to_int = (rgb_to_int << 8) + blueAverage;
  greyImage[id] = rgb_to_int;
  //greyImage[id].x = redAverage /numPixelsAveraged;
  //greyImage[id].y = greenAverage /numPixelsAveraged;
  //greyImage[id].z = blueAverage /numPixelsAveraged;
}	



void legoBlockCovert(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                     unsigned char* const d_greyImage, const size_t numRows, const size_t numCols,
                     const int legoBlockSize)
{
  const dim3 blockSize(32, 32, 1);  //TODO
  const dim3 gridSize(ceil(numCols/blockSize.x)+1,ceil(numRows/blockSize.y)+1);  
  

averageLegoBlockColor<<<gridSize, blockSize, sizeof(int) * numCols * numRows>>>(d_rgbaImage,
                   d_greyImage,
                   numRows, numCols, legoBlockSize);
				   
}



void cleanup() {

}
