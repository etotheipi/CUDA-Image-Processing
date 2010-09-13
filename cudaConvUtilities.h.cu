#ifndef _CUDA_CONV_UTILITIES_H_CU_
#define _CUDA_CONV_UTILITIES_H_CU_


#include <iostream>
#include <stdio.h>
#include <vector>
#include <cutil_inline.h>
#include <stopwatch.h>
#include "cudaImageHost.h"

#define IDX_1D(col, row, stride) ((col * stride) + row)
#define COL_2D(index, stride) (index / stride)
#define ROW_2D(index, stride) (index % stride)
#define ROUNDUP32(integer) ( ((integer-1)/32 + 1) * 32 )

#define SHMEM 8192
#define FLOAT_SZ sizeof(float)
#define INT_SZ   sizeof(int)

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//
// This macros is defined because EVERY convolution-like function has the same
// variables.  Mainly, the pixel identifiers for this thread based on block
// size, and the size of the padded rectangle that each block will work with
//
// ***This is actually the same as the CONVOLVE version
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_CONVOLUTION_VARIABLES(psfColRad, psfRowRad) \
\
   const int cornerCol = blockDim.x*blockIdx.x;   \
   const int cornerRow = blockDim.y*blockIdx.y;   \
   const int globalCol = cornerCol + threadIdx.x;   \
   const int globalRow = cornerRow + threadIdx.y;   \
   const int globalIdx = IDX_1D(globalCol, globalRow, imgRows);   \
\
   const int localCol    = threadIdx.x;   \
   const int localRow    = threadIdx.y;   \
   const int localIdx    = IDX_1D(localCol, localRow, blockDim.y);   \
   const int localPixels = blockDim.x*blockDim.y;   \
\
   const int padRectStride = blockDim.y + 2*psfRowRad;   \
   const int padRectCol    = localCol + psfColRad;   \
   const int padRectRow    = localRow + psfRowRad;   \
   const int padRectPixels = padRectStride * (blockDim.x + 2*psfColRad);   \
\
   __shared__ char sharedMem[8192]; \
   int* shmPadRect  = (int*)sharedMem;   \
   int* shmOutput   = (int*)&shmPadRect[ROUNDUP32(padRectPixels)];   \
   int nLoop;


////////////////////////////////////////////////////////////////////////////////
//
// Copy chunk of global memory to shared memory in this block for doing the
// convolution
//
////////////////////////////////////////////////////////////////////////////////
#define PREPARE_PADDED_RECTANGLE(psfColRad, psfRowRad) \
\
   nLoop = (padRectPixels/localPixels)+1;   \
   for(int loopIdx=0; loopIdx<nLoop; loopIdx++)   \
   {   \
      int prIndex = loopIdx*localPixels + localIdx;   \
      if(prIndex < padRectPixels)   \
      {   \
         int prCol = COL_2D(prIndex, padRectStride);   \
         int prRow = ROW_2D(prIndex, padRectStride);   \
         int glCol = cornerCol + prCol - psfColRad;   \
         int glRow = cornerRow + prRow - psfRowRad;   \
         int glIdx = IDX_1D(glCol, glRow, imgRows);   \
         if(glRow>=0 && glRow<imgRows && glCol>=0 && glCol<imgCols)   \
            shmPadRect[prIndex] = devInPtr[glIdx];   \
         else   \
            shmPadRect[prIndex] = 0; \
      }   \
   }   



////////////////////////////////////////////////////////////////////////////////
//
// We are using -1 as "OFF" and +1 as "ON" and 0 as "DONTCARE"
// The user is not expected to do this him/herself, and it's easy enough to 
// manipulate the data on the way in and out (just don't forget to convert back
// before copying out the result
//
////////////////////////////////////////////////////////////////////////////////
#define PREPARE_PADDED_RECTANGLE_MORPH(psfColRad, psfRowRad) \
\
   nLoop = (padRectPixels/localPixels)+1;   \
   for(int loopIdx=0; loopIdx<nLoop; loopIdx++)   \
   {   \
      int prIndex = loopIdx*localPixels + localIdx;   \
      if(prIndex < padRectPixels)   \
      {   \
         int prCol = COL_2D(prIndex, padRectStride);   \
         int prRow = ROW_2D(prIndex, padRectStride);   \
         int glCol = cornerCol + prCol - psfColRad;   \
         int glRow = cornerRow + prRow - psfRowRad;   \
         int glIdx = IDX_1D(glCol, glRow, imgRows);   \
         if(glRow>=0 && glRow<imgRows && glCol>=0 && glCol<imgCols)   \
            shmPadRect[prIndex] = devInPtr[glIdx]*2 - 1;   \
         else   \
            shmPadRect[prIndex] = -1; \
      }   \
   }   

////////////////////////////////////////////////////////////////////////////////
//
// Frequently, we want to pull some linear arrays into shared memory (usually 
// PSFs) which will be queried often, and we want them close to the threads.
//
// This macro temporarily reassigns all the threads to do the memory copy from
// global memory to shared memory in parallel.  Since the array may be bigger
// than the blocksize, some threads may be doing multiple mem copies
//
// ***This is actually the same as the FLOAT version
//
////////////////////////////////////////////////////////////////////////////////
#define COPY_LIN_ARRAY_TO_SHMEM(srcPtr, dstPtr, nValues) \
   nLoop = (nValues/localPixels)+1;   \
   for(int loopIdx=0; loopIdx<nLoop; loopIdx++)   \
   {   \
      int prIndex = loopIdx*localPixels + localIdx;   \
      if(prIndex < nValues)   \
      {   \
         dstPtr[prIndex] = srcPtr[prIndex]; \
      } \
   } 


////////////////////////////////////////////////////////////////////////////////
// This macro simply creates the declarations for the below kernels, to be
// used in the header file
////////////////////////////////////////////////////////////////////////////////
#define DECLARE_3X3_MORPH_KERNEL( name ) \
__global__ void  Morph3x3_##name##_Kernel(       \
               int*   devInPtr,          \
               int*   devOutPtr,          \
               int    imgCols,          \
               int    imgRows);


////////////////////////////////////////////////////////////////////////////////
//
// This macro creates optimized, unrolled versions of the generic
// morphological operation kernel for 3x3 structuring elements.
//
// Since it has no loops, and only one if-statement per thread, it should
// extremely fast.  The generic kernel is fast too, but slowed down slightly
// by the doubly-nested for-loops.
//
// TODO:  We should create 3x1 and 1x3 functions (and possibly Nx1 & 1xN)
//        so that we can further optimize morph ops for separable SEs
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_3X3_MORPH_KERNEL( name, seTargSum, \
                                            a00, a10, a20, \
                                            a01, a11, a21, \
                                            a02, a12, a22) \
__global__ void  Morph3x3_##name##_Kernel(       \
               int*   devInPtr,          \
               int*   devOutPtr,          \
               int    imgCols,          \
               int    imgRows)  \
{ \
   const int cornerCol = blockDim.x*blockIdx.x;   \
   const int cornerRow = blockDim.y*blockIdx.y;   \
   const int globalCol = cornerCol + threadIdx.x;   \
   const int globalRow = cornerRow + threadIdx.y;   \
   const int globalIdx = IDX_1D(globalCol, globalRow, imgRows);   \
\
   const int localCol    = threadIdx.x;   \
   const int localRow    = threadIdx.y;   \
   const int localIdx    = IDX_1D(localCol, localRow, blockDim.y);   \
   const int localPixels = blockDim.x*blockDim.y;   \
\
   const int padRectStride = blockDim.y + 2; \
   const int padRectCol    = localCol + 1;   \
   const int padRectRow    = localRow + 1;   \
\
   __shared__ char sharedMem[SHMEM];   \
   int* shmPadRect  = (int*)sharedMem;   \
   int* shmOutput   = (int*)&shmPadRect[512];   \
\
   shmOutput[localIdx] = -1;\
\
   int prIdx, prCol, prRow, glCol, glRow, glIdx; \
\
   prIdx = localIdx;   \
   prCol = COL_2D(prIdx, padRectStride);   \
   prRow = ROW_2D(prIdx, padRectStride);   \
   glCol = cornerCol + prCol - 1;   \
   glRow = cornerRow + prRow - 1;   \
   glIdx = IDX_1D(glCol, glRow, imgRows);   \
   if(glRow>=0 && glRow<imgRows && glCol>=0 && glCol<imgCols)   \
      shmPadRect[prIdx] = devInPtr[glIdx]*2 - 1;   \
   else   \
      shmPadRect[prIdx] = -1; \
\
   prIdx = localPixels + localIdx;   \
   prCol = COL_2D(prIdx, padRectStride);   \
   prRow = ROW_2D(prIdx, padRectStride);   \
   glCol = cornerCol + prCol - 1;   \
   glRow = cornerRow + prRow - 1;   \
   glIdx = IDX_1D(glCol, glRow, imgRows);   \
   if(glRow>=0 && glRow<imgRows && glCol>=0 && glCol<imgCols)   \
      shmPadRect[prIdx] = devInPtr[glIdx]*2 - 1;   \
   else   \
      shmPadRect[prIdx] = -1; \
\
   __syncthreads();   \
\
   int accum = 0;  \
   int coff, roff, seVal, shmPRCol, shmPRRow, shmPRIdx; \
\
   coff=-1; roff=-1; seVal = a00; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=-1; roff=0; seVal = a01; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=-1; roff=1; seVal = a02; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=0; roff=-1; seVal = a10; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=0; roff=0; seVal = a11; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=0; roff=1; seVal = a12; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=1; roff=-1; seVal = a20; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=1; roff=0; seVal = a21; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   coff=1; roff=1; seVal = a22; \
   shmPRCol = padRectCol + coff;     \
   shmPRRow = padRectRow + roff;     \
   shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);     \
   accum += seVal * shmPadRect[shmPRIdx];  \
\
   if(accum >= seTargSum)  \
      shmOutput[localIdx] = 1;  \
  \
   __syncthreads();     \
  \
   devOutPtr[globalIdx] = (shmOutput[localIdx] + 1) / 2;  \
}



////////////////////////////////////////////////////////////////////////////////
// CPU timer pretty much measures real time (wall clock time).  GPU timer 
// measures based on the number of GPU clock cycles, which is useful for 
// benchmarking memory copies and GFLOPs, but not wall time.
void   cpuStartTimer(void);
float  cpuStopTimer(void);
void   gpuStartTimer(void);
float  gpuStopTimer(void);

////////////////////////////////////////////////////////////////////////////////
// Read/Write images from/to files
void ReadFile(string fn, int* targPtr, int nCols, int nRows);

////////////////////////////////////////////////////////////////////////////////
// Writing file in space-separated format
void WriteFile(string fn, int* srcPtr, int nCols, int nRows);

////////////////////////////////////////////////////////////////////////////////
// Writing image to stdout
void PrintArray(int* srcPtr, int nCols, int nRows);

////////////////////////////////////////////////////////////////////////////////
// Copy a 3D texture from a host (float*) array to a device cudaArray
// The extent should be specified with all dimensions in units of *elements*
void prepareCudaTexture(float* h_src, 
                        cudaArray *d_dst,
                        cudaExtent const texExtent);
 

// Assume target memory has already been allocated, nPixels is odd
void createGaussian1D(float* targPtr, 
                      int    nPixels, 
                      float  sigma, 
                      float  ctr);

// Assume target memory has already been allocated, nPixels is odd
// Use col-row (D00_UL_ES)
void createGaussian2D(float* targPtr, 
                      int    nPixelsCol,
                      int    nPixelsRow,
                      float  sigmaCol,
                      float  sigmaRow,
                      float  ctrCol,
                      float  ctrRow);


// Assume diameter^2 target memory has already been allocated
// This filter is used for edge detection.  We always assume 
// square and symmetric kernels for LoG/edge detection, which
// is why there are no options for different dimensions
void createLaplacianOfGaussianKernel(float* targPtr,
                                     int    diameter);

// Assume diameter^2 target memory has already been allocated
int createBinaryCircle(int* targPtr,
                       int  diameter);

// Assume diameter^2 target memory has already been allocated
cudaImageHost createBinaryCircle( int diameter);




////////////////////////////////////////////////////////////////////////////////
// BASIC UNARY & BINARY *MASK* OPERATORS
// 
// Could create LUTs, but I'm not sure the extra implementation complexity
// actually provides much benefit.  These ops already run on the order of
// microseconds.
//
// NOTE:  These operators are for images with {0,1}, only the MORPHOLOGICAL
//        operators will operate with {-1,0,1}
//
////////////////////////////////////////////////////////////////////////////////
__global__ void  Mask_Union_Kernel(     int* srcA, int* srcB, int* dst);
__global__ void  Mask_Intersect_Kernel( int* srcA, int* srcB, int* dst);
__global__ void  Mask_Subtract_Kernel(  int* srcA, int* srcB, int* dst);
__global__ void  Mask_Difference_Kernel( int* srcA, int* srcB, int* dst);
__global__ void  Mask_Invert_Kernel(    int* srcA,            int* dst);
__global__ void  Mask_Copy_Kernel(      int* srcA,            int* dst);

// Yes!  You need two EXTRA buffers to use Image_Sum;
__global__ void  Image_SumReduceStep_Kernel(int* bufIn, int* bufOut, int lastBlockSize);
int Image_Sum(int* devPtr, int* devTemp1, int* devTemp2, int arraySize);


#endif
