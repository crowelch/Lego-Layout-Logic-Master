#include "utils.h"
#include "stdafx.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <tchar.h>
#include <thrust/device_vector.h>

using namespace std;

struct block {
	int color ;
	int bTNum ;
	int amount;
};



__device__
	bool inArray(int block[], int size, int numOfElem)
{

	for (int i = 0; i < numOfElem ; ++i)
	{
		if (block[i] >= size)
		{
			return false;
		}
	}
	__syncthreads();
	return true;
}

__device__
	bool equivalentBlockOnebyN(int *block, int row, int col, int blC, int blR, int **str, bool isRowCompare)
{
	if (isRowCompare)
	{
		for (int b = 0; b < blC; b++)
		{
			if (str[row][b] != str[row][col])
			{
				return false;
			}
		}
	}
	else
	{
		for (int b = 0; b < blR; b++)
		{
			if (str[b][col] != str[row][col])
			{
				return false;
			}
		}
	}
	return true;
}

__device__
	bool notInQueue(int *queue, int *block, int row, int col, int blC, int blR,  bool isRowCompare)
{
	if (isRowCompare)
	{
		for (int b = 0; b < blC; b++)
		{
			int oneDArr =  row + block[b] ;
			if (queue[oneDArr] == 1)
			{
				return true;
			}
		}
	}
	else
	{
		for (int b = 0; b < blR; b++)
		{
			int oneDArr =  row * block[b] + col ;
			if (queue[oneDArr]== 1)
			{
				return true;
			}
		}
	}
	return false;
}


__device__
	void addToQueue(int  *queue, int *block, int row, int col,  int blC, int blR, bool isRowCompare)
{
	if (isRowCompare)
	{
		for (int b = 0; b < blC; b++)
		{
			int oneDArr =  row * block[b] ;
			atomicAdd(&queue[oneDArr],   1);
		}
	}
	else
	{
		for (int b = 0; b < blR; b++)
		{
			int oneDArr =  row * block[b] + col ;
			atomicAdd(&queue[oneDArr],   1);
		}

	}
}

__device__
	bool isInQueue ( int *queue, int item ) { 
		if ( queue[item] == 1 ) {
			return true; 
		}
		else{
			return false; 
		}
} 


__global__
	void general(int *queue, int **str, int rowP, int colP, int blR, int blC, int **blockType,  int *colors, block *blockStruct)
{
	extern __shared__ int shared[];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	bool truth = false;
	bool badRow = false;
	bool badCol = false; 
	int blockC[4] ;
	int blockR[4] ;
	int oneDArr =  row * rowP + col;

	if (row >= rowP || col >= colP){  return; } 

	if (!isInQueue(queue, oneDArr) )
	{
		for (int bl = 0; bl < blC; bl++)
		{
			if (bl < colP)
			{
				blockC[bl] = col + bl;
			}
			else
			{
				badCol = true;
			}

		}
		for (int bl = 0; bl < blR; bl++)
		{
			if (bl < rowP)
			{
				blockR[bl] = row + bl;
			}
			else
			{
				badRow = true;
			}
		}
		__syncthreads();

		if (inArray(blockC, col, blC))
		{
			if (equivalentBlockOnebyN(blockC, row, col,  blC, blR, str, true))
			{
				if (!notInQueue(queue, blockC, row, col, blC, blR, true) & !badCol)
				{
					addToQueue(queue, blockC, row, col, blC, blR, true);
					truth = true;
					__syncthreads();
				}
			}
		}
		if (inArray(blockR, row, blR))
		{
			if (equivalentBlockOnebyN(blockR, row, col,blC, blR, str, false))
			{
				if (!notInQueue(queue, blockR, row, col, blC, blR,  false) & !badRow)
				{
					addToQueue(queue, blockR, row, col,  blC, blR, false);
					truth = true;
					__syncthreads();

				}
			}
		}

		if (truth == true)
		{
			for ( int i = 0 ; i < 5; i++)
			{
				if (blockStruct[i].color == colors[str[row][col]]) 
				{ 
					atomicAdd(&blockStruct[i].amount, 1 ) ;
				}

			}
		}
	}
}



void optimize(int str[][], int rows, int cols)  
{	
	int n = 1024;

	block block1x4[5];
	block block1x3[5];
	block block1x2[5];
	block block1x1[5];

	for (int i = 0;  i < 5; i++)
	{
			block1x4[i].color = i;
			block1x4[i].bTNum = 0;
			block1x4[i].amount = 0 ; 

			block1x3[i].color = i;
			block1x3[i].bTNum = 1;
			block1x3[i].amount = 0 ;

			block1x2[i].color = i;
			block1x2[i].bTNum = 2;
			block1x2[i].amount = 0 ;

			block1x1[i].color = i;
			block1x1[i].bTNum = 3;
			block1x1[i].amount = 0 ;
	}

	const dim3 block_dim(((rows * cols) + n - 1) / n);
	const dim3 thread_dim(n);

	char *colors[] =  { "black", "white", "purple", "yellow", "blue" } ;
	char *bT[] = {  "1x4", "1x3", "1x2", "1x1" };

	int  colorsNum[] = { 0, 1, 2, 3, 4 };
	int bTNum[] = { 0, 1, 2, 3, 4, 5, 6 } ;



	block *dBlock1x4, *dBlock1x3, *dBlock1x2,  *dBlock1x1;
	checkCudaErrors(cudaMalloc((void **)&dBlock1x4, sizeof(block) * 5)); 
	checkCudaErrors(cudaMalloc(&dBlock1x3, sizeof(block) * 5)); 
	checkCudaErrors(cudaMalloc(&dBlock1x2, sizeof(block) * 5)); 
	checkCudaErrors(cudaMalloc(&dBlock1x1, sizeof(block) * 5)); 
	checkCudaErrors(cudaMemcpy(dBlock1x4, block1x4,  sizeof(block) * 5, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dBlock1x3, block1x3,  sizeof(block) * 5, cudaMemcpyHostToDevice));	
	checkCudaErrors(cudaMemcpy(dBlock1x2, block1x2,  sizeof(block) * 5, cudaMemcpyHostToDevice));	
	checkCudaErrors(cudaMemcpy(dBlock1x1, block1x1,  sizeof(block) * 5, cudaMemcpyHostToDevice));

	int **dStr,  *dqueue, *dColorsNum, **dbTNum;
	checkCudaErrors(cudaMalloc(&dColorsNum, sizeof(int) * 5));  
	checkCudaErrors(cudaMalloc(&dbTNum, sizeof(int) * 7)); 
	checkCudaErrors(cudaMalloc(&dStr, sizeof(int) * (rows* cols) )); 
	checkCudaErrors(cudaMalloc(&dqueue, sizeof(int) * (rows * cols )));

	checkCudaErrors(cudaMemcpy(dColorsNum, colorsNum,  sizeof(int) *  5, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dbTNum, bTNum,  sizeof(int) * 7, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dStr, str,  sizeof(int) * (rows* cols), cudaMemcpyHostToDevice));

	general<<<block_dim, thread_dim, sizeof(int) * 72>>>(dqueue, dStr, rows, cols, 4, 4, dbTNum,  dColorsNum, dBlock1x4);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	general<<<block_dim, thread_dim, sizeof(int) * 72>>>(dqueue, dStr, rows, cols, 3, 3, dbTNum,  dColorsNum, dBlock1x3);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	general<<<block_dim, thread_dim, sizeof(int) * 72>>>(dqueue, dStr, rows, cols, 2, 2, dbTNum,  dColorsNum, dBlock1x2);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	general<<<block_dim, thread_dim, sizeof(int) * 72>>>(dqueue, dStr, rows, cols, 1, 1, dbTNum,  dColorsNum, dBlock1x1);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(block1x4, dBlock1x4, sizeof(block) * 5, cudaMemcpyDeviceToHost)); 
	checkCudaErrors(cudaMemcpy(block1x3, dBlock1x3, sizeof(block) * 5, cudaMemcpyDeviceToHost)); 
	checkCudaErrors(cudaMemcpy(block1x2, dBlock1x2, sizeof(block) * 5, cudaMemcpyDeviceToHost)); 
	checkCudaErrors(cudaMemcpy(block1x1, dBlock1x1, sizeof(block) * 5, cudaMemcpyDeviceToHost)); 

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	 
	int bcount = 0 ;
	for (int i = 0 ; i  < 5; i++)
	{                     
		printf ("Block Color: %c     Block Type: %c   Amount: %d \n",  colors[block1x4[i].color] , bT[block1x4[i].bTNum], block1x4[i].amount );	
	 	printf ("Block Color: %c     Block Type: %c   Amount: %d \n",  colors[block1x3[i].color] , bT[block1x3[i].bTNum], block1x3[i].amount );	
	 	printf ("Block Color: %c     Block Type: %c   Amount: %d \n",  colors[block1x2[i].color] , bT[block1x2[i].bTNum], block1x2[i].amount );	
	 	printf ("Block Color: %c     Block Type: %c   Amount: %d \n",  colors[block1x1[i].color] , bT[block1x1[i].bTNum], block1x1[i].amount );	
	   bcount = bcount +  block1x1[i].amount + block1x2[i].amount + block1x3[i].amount  + block1x4[i].amount ; 
	}
	printf ("Total Blocks: %d", bcount) ;      
}