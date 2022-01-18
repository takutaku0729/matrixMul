/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling
 * approach. It has been written for clarity of exposition to illustrate various
 * CUDA programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication. See also: V. Volkov and
 * J. Demmel, "Benchmarking GPUs to tune dense linear algebra," in Proc. 2008
 * ACM/IEEE Conf. on Supercomputing (SC '08), Piscataway, NJ: IEEE Press, 2008,
 * pp. Art. 31:1-11.
 */

// System includes
#include <assert.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fstream>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include <random>

#define INF INFINITY
#define BLOCK_SIZE 32
#define BLOCK_NUM 4
#define INFP 25
#define ZEROP 50
#define MODE 4   // tropical = 1, else = 2, infskip = 3, zeroskip = 4
#define ADD_MODE 1 //1:min-plus 2:max-plus

//define for switching debug mode

//#define DEBUG_COUNT
//#define SINGLE

//zero mode switch
#define ZEROTILE

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */

__global__ void MatrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void MinPlusTrop(float* C, float* A, float* B, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub = (Csub <= (As[ty][k] + Bs[k][tx]) ? Csub : As[ty][k] + Bs[k][tx]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the blocsub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void MinPlusTropSkip(float* C, float* A, float* B, float* infA, float* infB, int wA, int wB ,int* skipcounter) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = INF;

    int indexConstA = wA * ty + tx;
    int indexConstB = wB * ty + tx;

    int infIndexConstA = (wA / BLOCK_SIZE) * by;
    int infIndexConstB = wB / BLOCK_SIZE;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, infIndexA = infIndexConstA, infIndexB = bx; a <= aEnd; a += aStep, b += bStep, infIndexA++, infIndexB += infIndexConstB) {


#ifdef DEBUG_COUNT

        if (tx == 0 && ty == 0) {
            atomicAdd(&skipcounter[1], 1);
            if ((isinf(infA[infIndexA]) == 1) || (isinf(infB[infIndexB]) == 1)) {
                atomicAdd(&skipcounter[0], 1);
            }
        }

#endif

        //skip execution
        if ((isinf(infA[infIndexA]) == 1) || (isinf(infB[infIndexB]) == 1)) {
            continue;
        }

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + indexConstA];
        Bs[ty][tx] = B[b + indexConstB];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub = (Csub <= (As[ty][k] + Bs[k][tx]) ? Csub : As[ty][k] + Bs[k][tx]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the blocsub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + indexConstB] = Csub;
}

__global__ void MinPlusTropZeroSkip1(float* C, float* A, float* B, float* infA, float* infB, int wA, int wB, int* skipcounter) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = INF;

    int indexConstA = wA * ty + tx;
    int indexConstB = wB * ty + tx;

    int infIndexConstA = (wA / BLOCK_SIZE) * by;
    int infIndexConstB = wB / BLOCK_SIZE;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, infIndexA = infIndexConstA, infIndexB = bx; a <= aEnd; a += aStep, b += bStep, infIndexA++, infIndexB += infIndexConstB) {


#ifdef DEBUG_COUNT

        if (tx == 0 && ty == 0) {
            atomicAdd(&skipcounter[1], 1);
            if ((isinf(infA[infIndexA]) == 1) || (isinf(infB[infIndexB]) == 1)) {
                atomicAdd(&skipcounter[0], 1);
            }
        }

#endif

        //skip execution
        if ((isinf(infA[infIndexA]) == 1) || (isinf(infB[infIndexB]) == 1)) {
            continue;
        }

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + indexConstA];
        Bs[ty][tx] = B[b + indexConstB];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            if (Csub >= (As[ty][k] + Bs[k][tx])) {
                Csub = As[ty][k] + Bs[k][tx];
                    if (Csub == 0) {
                        break;
                    }
            }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the blocsub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + indexConstB] = Csub;

}

__global__ void MinPlusTropZeroSkip2(float* C, float* A, float* B, float* infA, float* infB, int wA, int wB, int* skipcounter) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = INF;

    int indexConstA = wA * ty + tx;
    int indexConstB = wB * ty + tx;

    int infIndexConstA = (wA / BLOCK_SIZE) * by;
    int infIndexConstB = wB / BLOCK_SIZE;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, infIndexA = infIndexConstA, infIndexB = bx; a <= aEnd; a += aStep, b += bStep, infIndexA++, infIndexB += infIndexConstB) {


#ifdef DEBUG_COUNT

        if (tx == 0 && ty == 0) {
            atomicAdd(&skipcounter[1], 1);
            if ((isinf(infA[infIndexA]) == 1) || (isinf(infB[infIndexB]) == 1)) {
                atomicAdd(&skipcounter[0], 1);
            }
        }

#endif

        //skip execution
        if ((infA[infIndexA] == 1) || (infB[infIndexB] == 1)) {
            continue;
        }

        if ((infA[infIndexA] == -1) && (infB[infIndexB] == -1)) {
            Csub = 0;
            break;
        }

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + indexConstA];
        Bs[ty][tx] = B[b + indexConstB];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            if (Csub >= (As[ty][k] + Bs[k][tx])) {
                Csub = As[ty][k] + Bs[k][tx];
                if (Csub == 0) {
                    break;
                }
            }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the blocsub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + indexConstB] = Csub;

}

__global__ void MaxPlusTrop(float* C, float* A, float* B, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub = (Csub >= (As[ty][k] + Bs[k][tx]) ? Csub : As[ty][k] + Bs[k][tx]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the blocsub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void MaxPlusTropSkip(float* C, float* A, float* B, float* infA, float* infB, int wA, int wB, int* skipcounter) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = INF;

    int indexConstA = wA * ty + tx;
    int indexConstB = wB * ty + tx;

    int infIndexConstA = (wA / BLOCK_SIZE) * by;
    int infIndexConstB = wB / BLOCK_SIZE;
    

    //skip execution
    for (int infIndexA = infIndexConstA, infIndexB =bx; infIndexA < infIndexConstA + (wA / BLOCK_SIZE); infIndexA++, infIndexB += infIndexConstB) {
        if ((isinf(infA[infIndexA]) == 1) || (isinf(infB[infIndexB]) == 1)) {
            int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
            C[c + indexConstB] = INF;


#ifdef DEBUG_COUNT

            if (tx == 0 && ty == 0) {
                atomicAdd(&skipcounter[1], 1);
                if ((isinf(infA[infIndexA]) == 1) || (isinf(infB[infIndexB]) == 1)) {
                    atomicAdd(&skipcounter[0], 1);
                }
            }

#endif


            return;
        }
    }

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, infIndexA = infIndexConstA, infIndexB = bx; a <= aEnd; a += aStep, b += bStep, infIndexA++, infIndexB += infIndexConstB) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + indexConstA];
        Bs[ty][tx] = B[b + indexConstB];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub = (Csub >= (As[ty][k] + Bs[k][tx]) ? Csub : As[ty][k] + Bs[k][tx]);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the blocsub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + indexConstB] = Csub;
}

__global__ void InfCheck(float* C, float* A, float* B, float* infA, float* infB, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the first sub-matrix of A processed by the block
    int bBegin = wB * BLOCK_SIZE * by;

    __shared__ int infcheckA;
    __shared__ int infcheckB;

    infcheckA = 0;
    infcheckB = 0;

    __syncthreads();


    if (isinf(A[aBegin + tx + ty * wA]) == 0) {
        infcheckA = 1;
    }

    if (isinf(B[bBegin + tx + ty * wB]) == 0) {
        infcheckB = 1;
    }

    __syncthreads();
    
    if (tx == 0 && ty == 0) {
        if (infcheckA == 0) {
            infA[bx + (wA / BLOCK_SIZE) * by] = 1;
        }
        if (infcheckB == 0) {
            infB[bx + (wB / BLOCK_SIZE) * by] = 1;
        }
    }
}

__global__ void ZeroCheck(float* C, float* A, float* B, float* infA, float* infB, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the first sub-matrix of A processed by the block
    int bBegin = wB * BLOCK_SIZE * by;

    __shared__ int zerocheckA;
    __shared__ int zerocheckB;

    zerocheckA = 0;
    zerocheckB = 0;

    __syncthreads();


    if (A[aBegin + tx + ty * wA] != 0) {
        zerocheckA = 1;
    }

    if (B[bBegin + tx + ty * wB] != 0) {
        zerocheckB = 1;
    }

    __syncthreads();

    if (tx == 0 && ty == 0) {
        if (zerocheckA == 0) {
            infA[bx + (wA / BLOCK_SIZE) * by] = -1;
        }
        if (zerocheckB == 0) {
            infB[bx + (wB / BLOCK_SIZE) * by] = -1;
        }
    }
}


void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; i++) {
    data[i] = val;
  }
}



void ConstantInitRand(float* data, int p, int pz, const dim3& size) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<> rand100(1, 100);
   int infcount = 0;
   int count = 0;
   for (int y = 0; y < size.y / BLOCK_SIZE; y++) {
       for (int x = 0; x < size.x / BLOCK_SIZE; x++) {
           count++;
           if (rand100(rnd) < p) {
               for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                   for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                       data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = INF;
                   }
               }
               infcount++;
           }
           else {
#ifdef ZEROTILE
               if (rand100(rnd) < pz) {
                   for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                       for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 0;
                       }
                   }
               }
               else {
                   for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                       for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 1;
                       }
                   }
               }
#else
               for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                   for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                       if (rand100(rnd) < pz) {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 0;
                       }
                       else {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 1;
                       }
                   }
               }
#endif
           }
       }
   }

   //printf("inf percentage : %d / %d\n", infcount, count);

   /* print data
    int ex;
    for (int a = 0; a < size.y; a++) {
        for (int b = 0; b < size.x; b++) {
            ex = int(data[a * size.y + b]);
            printf("%d ", ex);
        }
        printf("\n");
    }
    */
}



void InfInit(float* data, int size, float val) {
    for (int i = 0; i < size; i++) {
        data[i] = val;
    }
}

void SetFileData(float* data, int size) {
    std::ifstream ifs("USA-road-d.NY.gr");

    if (!ifs) {
        std::cout << "Error: file not opened." << std::endl;
    }

    std::string tmp;
    for (int i = 0; i < 7; i++) {
        std::getline(ifs, tmp);
    }

    std::string buf;
    int x,y,value;
    for (int i = 0; i < size*size; i++) {
        ifs >> buf >> x >> y >> value;
        if (x < size && y < size) {
            data[x + y * size] = value;
        }
    }

    ifs.close();
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char** argv, int block_size, const dim3& dimsA,
    const dim3& dimsB, std::ofstream& writing_file, int mode, int addmode) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  unsigned int size_infA = (dimsA.x/BLOCK_SIZE) * (dimsA.y/BLOCK_SIZE);
  unsigned int mem_size_infA = sizeof(float) * size_infA;
  float *h_infA;
  checkCudaErrors(cudaMallocHost(&h_infA, mem_size_infA));
  unsigned int size_infB = (dimsB.x / BLOCK_SIZE) * (dimsB.y / BLOCK_SIZE);
  unsigned int mem_size_infB = sizeof(float) * size_infA;
  float *h_infB;
  checkCudaErrors(cudaMallocHost(&h_infB, mem_size_infB));
  int *skipcounter;
  checkCudaErrors(cudaMallocHost(&skipcounter, sizeof(int)*2));
  
  cudaStream_t stream;

  // Initialize host memory

  /*
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);
  */

  int InfP = INFP;  //set inf percentage
  int ZeroP = ZEROP;
  
  printf("set data of matrix A\n");
  ConstantInitRand(h_A, InfP, ZeroP, dimsA);   //set random (sparce) data to host memory
  printf("set data of matrix B\n");
  ConstantInitRand(h_B, InfP, ZeroP, dimsB);


  InfInit(h_infA, size_infA, 0);  //init inf check matrix
  InfInit(h_infB, size_infB, 0);

  skipcounter[0] = 0;   //for debug
  skipcounter[1] = 0;

  /*
  SetFileData(h_A, dimsA.x);    //set file data to host memory
  SetFileData(h_B, dimsB.x);
  */

  // Allocate device memory
  float* d_A, * d_B, * d_C, * inf_A, * inf_B;
  int* d_skipcounter;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&inf_A), mem_size_infA));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&inf_B), mem_size_infB));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_skipcounter), sizeof(int)*2));
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t startA, stopA, startB, stopB;
  checkCudaErrors(cudaEventCreate(&startA));
  checkCudaErrors(cudaEventCreate(&stopA));
  checkCudaErrors(cudaEventCreate(&startB));
  checkCudaErrors(cudaEventCreate(&stopB));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // copy host memory to device
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(inf_A, h_infA, mem_size_infA, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(inf_B, h_infB, mem_size_infB, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_skipcounter, skipcounter, sizeof(int)*2, cudaMemcpyHostToDevice, stream));

  // Setup execution parameters
  dim3 block(block_size, block_size);
  dim3 grid(dimsB.x / block.x, dimsA.y / block.y);

  // Create and start timer
  //printf("Computing result using CUDA Kernel...\n");

  checkCudaErrors(cudaStreamSynchronize(stream));

  // Execute the kernel
  int nIter = 1;

  //printf("mode : %d \n", mode);

  
  if (addmode == 1) {
      if (mode == 1) {
          checkCudaErrors(cudaEventRecord(startA, stream));
          for (int j = 0; j < nIter; j++) {
              MinPlusTrop
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
          }
      }
      else if (mode == 2) {
          checkCudaErrors(cudaEventRecord(startA, stream));
          for (int j = 0; j < nIter; j++) {
              MatrixMulCUDA
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
          }
      }
      else if (mode == 3) {
          for (int j = 0; j < nIter; j++) {
              checkCudaErrors(cudaEventRecord(startB, stream));
              InfCheck
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, inf_A, inf_B, dimsA.x, dimsB.x);
              checkCudaErrors(cudaDeviceSynchronize());
              checkCudaErrors(cudaEventRecord(stopB, stream));
              checkCudaErrors(cudaEventRecord(startA, stream));
              MinPlusTropSkip
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, inf_A, inf_B, dimsA.x, dimsB.x, d_skipcounter);
          }
      }
      else{
          for (int j = 0; j < nIter; j++) {
              checkCudaErrors(cudaEventRecord(startB, stream));
              InfCheck
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, inf_A, inf_B, dimsA.x, dimsB.x);
              checkCudaErrors(cudaDeviceSynchronize());
              ZeroCheck
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, inf_A, inf_B, dimsA.x, dimsB.x);
              checkCudaErrors(cudaDeviceSynchronize());
              checkCudaErrors(cudaEventRecord(stopB, stream));
              checkCudaErrors(cudaEventRecord(startA, stream));
              MinPlusTropZeroSkip2
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, inf_A, inf_B, dimsA.x, dimsB.x, d_skipcounter);
          }
      }
  }
  else {
      if (mode == 1) {
          checkCudaErrors(cudaEventRecord(startA, stream));
          for (int j = 0; j < nIter; j++) {
              MaxPlusTrop
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
          }
      }
      else if (mode == 2) {
          checkCudaErrors(cudaEventRecord(startA, stream));
          for (int j = 0; j < nIter; j++) {
              MatrixMulCUDA
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
          }
      }
      else if (mode == 3) {
          checkCudaErrors(cudaEventRecord(startA, stream));
          for (int j = 0; j < nIter; j++) {
              checkCudaErrors(cudaEventRecord(startB, stream));
              InfCheck
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, inf_A, inf_B, dimsA.x, dimsB.x);
              checkCudaErrors(cudaDeviceSynchronize());
              checkCudaErrors(cudaEventRecord(stopB, stream));
              checkCudaErrors(cudaEventRecord(stopB, stream));
              MaxPlusTropSkip
                  << <grid, block, 0, stream >> > (d_C, d_A, d_B, inf_A, inf_B, dimsA.x, dimsB.x, d_skipcounter);
          }
      }
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stopA, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stopA));

  float msecTotalA = 0.0f;
  float msecTotalB = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotalA, startA, stopA));

  float msecTotal = msecTotalA;

 
  if (mode == 3) {
      checkCudaErrors(cudaEventElapsedTime(&msecTotalB, startB, stopB));
      msecTotal += msecTotalB;
  }

  // Compute and print the performance
  float msecPerMatrixMul = (msecTotal) / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time = %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecTotalA, flopsPerMatrixMul, block.x * block.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(
      cudaMemcpyAsync(h_infA, inf_A, mem_size_infA, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(
      cudaMemcpyAsync(skipcounter, d_skipcounter, sizeof(int)*2, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));


  //file output

#ifdef DEBUG_COUNT

  writing_file << "," + std::to_string(skipcounter[0] / nIter);

#else

  writing_file << "," + std::to_string(gigaFlops);

#endif

#ifdef DEBUG_COUNT
  if(mode == 3){
          printf("skipcount = %d / %d\n\n", skipcounter[0] / nIter, skipcounter[1] / nIter);
  }

#endif

  /*
  printf("Checking computed result for correctness: ");
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }*/

  //printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaFree(inf_B));
  checkCudaErrors(cudaFree(inf_A));
  checkCudaErrors(cudaFreeHost(skipcounter));
  checkCudaErrors(cudaFree(d_skipcounter));
  checkCudaErrors(cudaEventDestroy(startA));
  checkCudaErrors(cudaEventDestroy(stopA));
  checkCudaErrors(cudaEventDestroy(startB));
  checkCudaErrors(cudaEventDestroy(stopB));


  /*
  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
  */
}

/**
 * Program main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  /*
  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf(
        "  Note: Outer matrix dimensions of A & B matrices"
        " must be equal.\n");

    exit(EXIT_SUCCESS);
  }
  */

  int n;    //デバイス数
  checkCudaErrors(cudaGetDeviceCount(&n));

  /*
  for (int i = 0; i < n; ++i) {
      cudaDeviceProp dev;

      // デバイスプロパティ取得
      checkCudaErrors(cudaGetDeviceProperties(&dev, i));

      printf("device %d\n", i);
      printf(" device name : %s\n", dev.name);
      printf(" total global memory : %d (MB)\n", dev.totalGlobalMem / 1024 / 1024);
      printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock / 1024);
      printf(" register / block : %d\n", dev.regsPerBlock);
      printf(" warp size : %d\n", dev.warpSize);
      printf(" max pitch : %d (B)\n", dev.memPitch);
      printf(" max threads / block : %d\n", dev.maxThreadsPerBlock);
      printf(" max size of each dim. of block : (%d, %d, %d)\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
      printf(" max size of each dim. of grid  : (%d, %d, %d)\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
      printf(" clock rate : %d (MHz)\n", dev.clockRate / 1000);
      printf(" total constant memory : %d (KB)\n", dev.totalConstMem / 1024);
      printf(" compute capability : %d.%d\n", dev.major, dev.minor);
      printf(" alignment requirement for texture : %d\n", dev.textureAlignment);
      printf(" device overlap : %s\n", (dev.deviceOverlap ? "ok" : "not"));
      printf(" num. of multiprocessors : %d\n", dev.multiProcessorCount);
      printf(" kernel execution timeout : %s\n", (dev.kernelExecTimeoutEnabled ? "on" : "off"));
      printf(" integrated : %s\n", (dev.integrated ? "on" : "off"));
      printf(" host memory mapping : %s\n", (dev.canMapHostMemory ? "on" : "off"));

      printf(" compute mode : ");
      if (dev.computeMode == cudaComputeModeDefault) printf("default mode (multiple threads can use) \n");
      else if (dev.computeMode == cudaComputeModeExclusive) printf("exclusive mode (only one thread will be able to use)\n");
      else if (dev.computeMode == cudaComputeModeProhibited) printf("prohibited mode (no threads can use)\n");
     
  }*/

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = BLOCK_SIZE;

  int block_num = BLOCK_NUM;

  dim3 dimsA(block_num * block_size, block_num * block_size, 1);
  dim3 dimsB(block_num * block_size, block_num * block_size, 1);

  /*
  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // height of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }
  */

  int infp = INFP;

  std::ofstream writing_file;
  std::string filename = "inf" + std::to_string(infp) + ".csv";
  writing_file.open(filename, std::ios::app);

  int matrix_result = 0;

  int max_size = 8192;
  
  int avg_count = 25;

#ifdef DEBUG_COUNT

  printf("\n\nskipcount\n");


  for (int size = block_size; size <= max_size; size *= 2) {
      writing_file << "skip-" + std::to_string(size);
      dimsA.x = size;
      dimsA.y = size;
      dimsB.x = size;
      dimsB.y = size;
      printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
          dimsB.y);
      for (int i = 0; i <= avg_count; i++) {
          matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, writing_file, 3, ADD_MODE);
      }
      writing_file << "\n";
  }
  exit(matrix_result);
#endif



#ifdef SINGLE

  int size = block_size * block_num;
  dimsA.x = size;
  dimsA.y = size;
  dimsB.x = size;
  dimsB.y = size;
  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
      dimsB.y);
  matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, writing_file, MODE, ADD_MODE);
  exit(matrix_result);

#else
  /*
  printf("noskip\n");

  for (int size = block_size; size <= max_size; size *= 2) {
      writing_file << "noskip-" + std::to_string(size);
      dimsA.x = size;
      dimsA.y = size;
      dimsB.x = size;
      dimsB.y = size;
      printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
          dimsB.y);
      for (int i = 0; i <= avg_count; i++) {
          matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, writing_file, 1, ADD_MODE);
      }
      writing_file << "\n";
  }
  
  */
  printf("\n\nskip\n");


  for (int size = block_size; size <= max_size; size *= 2) {
      writing_file << "skip-" + std::to_string(size);
      dimsA.x = size;
      dimsA.y = size;
      dimsB.x = size;
      dimsB.y = size;
      printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
          dimsB.y);
      for (int i = 0; i <= avg_count; i++) {
          matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, writing_file, MODE, ADD_MODE);
      }
      writing_file << "\n";
  }
  exit(matrix_result);
#endif

}
