


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <cstdlib>




__global__
void averageBlocks(uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      uchar4* const outputImageRGBA,
                     int legoSize)
{
    int block_x = blockDim.x * blockIdx.x + threadIdx.x;
    int block_y = blockDim.y * blockIdx.y + threadIdx.y;
    int pixelX;
    int pixelY;
    int numPixelsAveraged = 0;
    int redAverage = 0; 
    int greenAverage = 0; 
    int blueAverage = 0; 
    //int max_x = blockDim.x * gridDim.x;
    for(int i = 0; i < legoSize; i++)
    {
        pixelY = (block_y *legoSize) + i;
       for(int j = 0; j < legoSize; j++)
       {
           pixelX = block_x * legoSize + j;
           int id = pixelX + (pixelY * numCols);
           if(id < numRows*numCols)
           {
                redAverage = redAverage + inputImageRGBA[id].x; 
                greenAverage = greenAverage + inputImageRGBA[id].y; 
                blueAverage = blueAverage + inputImageRGBA[id].z; 
                numPixelsAveraged ++;
           }
       }
     }
     redAverage = (redAverage / numPixelsAveraged);
     blueAverage = (blueAverage / numPixelsAveraged);
     greenAverage = (greenAverage / numPixelsAveraged);
     uchar4 outputPixel = make_uchar4(redAverage, greenAverage, blueAverage, 255);
    for(int i = 0; i < legoSize; i++)
    {
        pixelY = ((block_y*legoSize) + i);
       for(int j = 0; j < legoSize; j++)
       {
           pixelX = block_x * legoSize + j;
           int id = pixelX + pixelY * numCols;
           if(id < numRows*numCols)
           {
               printf ("RGB: %d %d %d\n", redAverage, greenAverage, blueAverage);
                inputImageRGBA[id] = outputPixel;
           }
       }
     }
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      uchar4* const outputImageRGBA,
                     int legoSize)
{
    const int baseLegoRed[9] =   {255,40 ,110,193,52 ,245,242,161,27};
    const int baseLegoGreen[9] = {0  ,127,153,223,43 ,205,243,165,42};
    const int baseLegoBlue[9] =  {0  ,70 ,201,240,117,47 ,242,162,52};
    int combined_x = blockDim.x * blockIdx.x + threadIdx.x;
    int combined_y = blockDim.y * blockIdx.y + threadIdx.y;
    //int max_x = blockDim.x * gridDim.x;
    int id = combined_x + combined_y * numCols;
    if(id < numRows*numCols){ 
        
   
        
      
    int leastDistance = 10000;
    int thisDistance;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    printf ("RGB: %d %d %d\n", red, green, blue);
    for(int i = 0; i < (sizeof(baseLegoRed)/sizeof(int)); i++)
    {
           thisDistance = ((abs(baseLegoRed[i] - inputImageRGBA[id].x) * abs(baseLegoRed[i] - inputImageRGBA[id].x)) + 
                           (abs(baseLegoGreen[i] - inputImageRGBA[id].y) *  abs(baseLegoGreen[i] - inputImageRGBA[id].y)) + 
                           (abs(baseLegoBlue[i] - inputImageRGBA[id].z) * abs(baseLegoBlue[i] - inputImageRGBA[id].z)));
          

          if(thisDistance < leastDistance)
          {
                  leastDistance = thisDistance;
                  red = baseLegoRed[i];
                  green = baseLegoGreen[i];
                  blue = baseLegoBlue[i];
          }
        
        }
      uchar4 outputPixel = make_uchar4(red, green, blue, 255);
      outputImageRGBA[id] = outputPixel;
      }
}




void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{



}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  int legoSize = 5;
  const dim3 blockSize(32, 32, 1);  //TODO
  const dim3 gridSize(ceil(numCols/blockSize.x)+1,ceil(numRows/blockSize.y)+1);    
   int numBlockCols = (numCols/legoSize) + 1;
  int numBlockRows = (numRows/legoSize) + 1;
  const dim3 blockSize2(32, 32, 1); 
  const dim3 gridSize2(ceil(numBlockCols/blockSize2.x),ceil(numBlockRows/blockSize2.y));  
   averageBlocks<<<gridSize2, blockSize2>>>(d_inputImageRGBA,
                      numRows,
                      numCols,
                      d_outputImageRGBA,
                      legoSize);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
      separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                      numRows,
                      numCols,
                      d_outputImageRGBA,
                      legoSize
                     );


}



void cleanup() {

}
