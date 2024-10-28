#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
  float CValue = 0;

    int Row = blockIdx.y*TILE_SIZE + threadIdx.y;
    int Col = blockIdx.x*TILE_SIZE + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    for (int i = 0; i <  (k-1)/(float)TILE_SIZE+1; i++) {

         if (i*TILE_SIZE + threadIdx.x < k && Row < m)
             sA[threadIdx.y][threadIdx.x] = A[Row*k + i*TILE_SIZE + threadIdx.x];
         else
             sA[threadIdx.y][threadIdx.x] = 0;

         if (i*TILE_SIZE + threadIdx.y < k && Col < n)
             sB[threadIdx.y][threadIdx.x] = B[(i*TILE_SIZE + threadIdx.y)*n + Col];
         else
             sB[threadIdx.y][threadIdx.x] = 0;

         __syncthreads();

         for (int n = 0; n < TILE_SIZE; ++n)
             CValue += sA[threadIdx.y][n] * sB[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < m && Col < n)
        C[Row*n+Col] = CValue;
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((n-1)/BLOCK_SIZE+1,(m-1)/BLOCK_SIZE+1,1);
    dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,1);

    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
	mysgemm<<<DimGrid,DimBlock>>>(m,n,k,A,B,C);
    /*************************************************************************/
}


