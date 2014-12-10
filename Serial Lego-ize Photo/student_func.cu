


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <cstdlib>


int legoSize = 5;



//Note this code does not actually genrate the image just does the same operations
//and was used only to test the speed of the parallel algorithm

int *d_legoBlockPieces, *h_legoBlockPieces;


void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
        const int baseLegoRed[14] =   {245,218,196,205,13 ,40 ,204,242,27,52, 39, 35};
    const int baseLegoGreen[14] = {205,133,40 ,98 ,105,127,142,243,42,43, 70, 71};
    const int baseLegoBlue[14] =  {47 ,64 ,27 ,152,171,70 ,104,242,52,117,44, 139};
    h_legoBlockPieces = (int *)malloc(sizeof(int) * (numCols/legoSize) * (numCols/legoSize));
    
   std::cout << numRows*numCols << std::endl; 
   
    int pixelX;
    int pixelY;
    int numPixelsAveraged = 0;
    int redAverage; 
    int greenAverage; 
    int blueAverage;
    for(int block_y = 0; block_y < (numRows/legoSize); block_y ++){
        for(int block_x = 0; block_x < (numCols/legoSize); block_x++){
              int numPixelsAveraged = 0;
              int redAverage = 0; 
              int greenAverage = 0; 
              int blueAverage = 0;
                  for(int i = 0; i < legoSize; i++)
                  {
                       pixelY = (block_y *legoSize) + i;
                       for(int j = 0; j < legoSize; j++)
                       {
                           pixelX = block_x * legoSize + j;
                           int id = pixelX + (pixelY * numCols);
                           if(id < numRows*numCols)
                           {
                                redAverage = redAverage + h_inputImageRGBA[id].x; 
                                greenAverage = greenAverage + h_inputImageRGBA[id].y; 
                                blueAverage = blueAverage + h_inputImageRGBA[id].z; 
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

           }
       }
     }
        }
    }
    for(int combined_y = 0; combined_y < (numRows/legoSize); combined_y ++){
       for(int combined_x = 0; combined_x < (numCols/legoSize); combined_x++){
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
           thisDistance = ((abs(baseLegoRed[i] - h_inputImageRGBA[id].x) * abs(baseLegoRed[i] - h_inputImageRGBA[id].x)) + 
                           (abs(baseLegoGreen[i] - h_inputImageRGBA[id].y) *  abs(baseLegoGreen[i] - h_inputImageRGBA[id].y)) + 
                           (abs(baseLegoBlue[i] - h_inputImageRGBA[id].z) * abs(baseLegoBlue[i] - h_inputImageRGBA[id].z)));
          
          if(thisDistance < leastDistance)
          {
                  leastDistance = thisDistance;
                  red = baseLegoRed[i];
                  green = baseLegoGreen[i];
                  blue = baseLegoBlue[i];
                  colorNum = i;
          }
        }
      h_legoBlockPieces[(combined_x/legoSize) + ((combined_y/legoSize) * (numCols/legoSize))] = colorNum;   
      uchar4 outputPixel = make_uchar4(red, green, blue, 255);

      }
           
           
       }
    }
       
        


}



void cleanup() {

}
