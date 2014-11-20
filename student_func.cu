//Lego Layout Logic Master
//Cuda Functions

#include "reference_calc.cpp"
#include "utils.h"

__global__
void averageLegoBlockColor(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols, const int legoSize)
{
  __shared__ int legoBlockNums[numRows * numCols];
  int absolute_image_position_x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int absolute_image_position_y = (blockDim.y * blockIdx.y) + threadIdx.y; 
  int id = absolute_image_position_x + (absolute_image_position_y * numCols);
  int legoRowNum = (absolute_image_position_x / legoSize) + 1;
  int legoColumnNum = (absolute_image_position_y / legoSize) + 1;
  int legoPieceId = legoRowNum + (legoColumnNum * (numCols/legoSize));
  blockNums[id] = legoPieceId;
  __syncthreads();
  float redAverage,greenAverage, blueAverage;
  int numPixelsAveraged;
  for(int i = 0; i < numRows*numCols; i++) //loop should find the pixel itself
  {
    if(legoBlockNums[i] == legoPieceId)
		{
		    redAverage = redAverage + inputChannel[i].x;
			greenAverage = greenAverage + inputChannel[i].y;
			greenAverage = greenAverage + inputChannel[i].z;
			numPixelsAveraged ++;
		}	
  }
  outputChannel[id].x = redAverage /numPixelsAveraged;
  outputChannel[id].y = greenAverage /numPixelsAveraged;
  outputChannel[id].z = blueAverage /numPixelsAveraged;
}	



void legoBlockCovert(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        const int legoBlockSize)
{
  const dim3 blockSize(32, 32, 1);  //TODO
  const dim3 gridSize(ceil(numCols/blockSize.x)+1,ceil(numRows/blockSize.y)+1);  
  

averageLegoBlockColor<<<gridSize, blockSize, sizeof(int) * numCols * numRows>>>(inputChannel,
                   outputChannel,
                   numRows, numCols, legoBlockSize);
				   
}



void cleanup() {

}
