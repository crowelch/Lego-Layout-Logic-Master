


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <cstdlib>


int legoSize = 5;


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
               //printf ("RGB: %d %d %d\n", redAverage, greenAverage, blueAverage);
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
                     int legoSize,
                     int * legoPieceGrid)
{ //24 Bright Yellow,106 Bright Orange, 21 Bright Red, Bright purple
  //23 Bright Blue, Dark Green, Bright Green-Yellow, Red-Brown,Nougat
  //1 White, 26 Black,154 Dark Red,268 Medium Lilac, 140 Earth Blue,141 Earth GReen  
    //const int baseLegoRed[11] =   {245,218,196,205,13 ,40 ,164,105,204,242,27};
    //const int baseLegoGreen[11] = {205,133,40 ,98 ,105,127,189,64 ,142,243,42};
    //const int baseLegoBlue[11] =  {47 ,64 ,27 ,152,171,70 ,70 ,39 ,104,242,52};
    const int baseLegoRed[14] =   {245,218,196,205,13 ,40 ,204,242,27,52, 39, 35};
    const int baseLegoGreen[14] = {205,133,40 ,98 ,105,127,142,243,42,43, 70, 71};
    const int baseLegoBlue[14] =  {47 ,64 ,27 ,152,171,70 ,104,242,52,117,44, 139};
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
    unsigned int colorNum ;  
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
                  colorNum = i;
          }
        }
      legoPieceGrid[(combined_x/legoSize) + ((combined_y/legoSize) * (numCols/legoSize))] = colorNum;   
      uchar4 outputPixel = make_uchar4(red, green, blue, 255);
      outputImageRGBA[id] = outputPixel;
      }
}



int *d_legoBlockPieces, *h_legoBlockPieces;
void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
    int numBlockCols = (numColsImage/legoSize)+1;
    int numBlockRows = (numRowsImage/legoSize)+1;
    checkCudaErrors(cudaMalloc(&d_legoBlockPieces,   sizeof(int) * numBlockCols * numBlockRows));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  int numBlockCols = (numCols/legoSize);
  int numBlockRows = (numRows/legoSize);
  const dim3 blockSize(32, 32, 1);  //TODO
  const dim3 gridSize(ceil(numCols/blockSize.x)+1,ceil(numRows/blockSize.y)+1);    
  const dim3 blockSize2(32, 32, 1); 
  const dim3 gridSize2((numBlockCols/blockSize2.x)+1,(numBlockRows/blockSize2.y)+1);  
  //constc dim3 gridSize2(ceil(numBlockCols/blockSize2.x),ceil(numBlockRows/blockSize2.y));  
   averageBlocks<<<gridSize2, blockSize2>>>(d_inputImageRGBA,
                      numRows,
                      numCols,
                      d_outputImageRGBA,
                      legoSize);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    h_legoBlockPieces = (int *)malloc(sizeof(int) * numBlockCols * numBlockRows);
      separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                      numRows,
                      numCols,
                      d_outputImageRGBA,
                      legoSize,
                      d_legoBlockPieces                         
                     );
    cudaMemcpy(h_legoBlockPieces, d_legoBlockPieces, sizeof(int) * numBlockCols * numBlockRows, cudaMemcpyDeviceToHost);
    int deviceResult[numBlockCols][numBlockRows];
    int counter = 0;
    for(int y = 0; y < numBlockRows; y++)
    {
       for(int x = 0; x < numBlockCols; x++) 
       {
           deviceResult[x][y] = h_legoBlockPieces[counter];
           //std::cout << "x: " << x << " y: " << y << " color ID " << h_legoBlockPieces[counter] << std::endl;
           counter++;
       }
    }
	
	optimize(deviceResult,numBlockRows,numBlockCols);
        


}



