using namespace std;

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "cudaConvUtilities.h.cu"
using namespace std;

unsigned int cpuTimerVariable;
cudaEvent_t eventTimerStart;
cudaEvent_t eventTimerStop;

// Assume target memory has already been allocated, nPixels is odd
void createGaussian1D(float* targPtr, 
                      int    nPixels, 
                      float  sigma, 
                      float  ctr)
{
   if(nPixels%2 != 1)
   {
      cout << "***Warning: createGaussian(...) only defined for odd pixel"  << endl;
      cout << "            dimensions.  Undefined behavior for even sizes." << endl;
   }

   float pxCtr = (float)(nPixels/2 + ctr);   
   float sigmaSq = sigma*sigma;
   float denom = sqrt(2*M_PI*sigmaSq);
   float dist;
   for(int i=0; i<nPixels; i++)
   {
      dist = (float)i - pxCtr;
      targPtr[i] = exp(-0.5 * dist * dist / sigmaSq) / denom;
   }
}

// Assume target memory has already been allocate, nPixels is odd
// Use col-row (D00_UL_ES)
void createGaussian2D(float* targPtr, 
                      int    nPixelsCol,
                      int    nPixelsRow,
                      float  sigmaCol,
                      float  sigmaRow,
                      float  ctrCol,
                      float  ctrRow)
{
   if(nPixelsCol%2 != 1 || nPixelsRow != 1)
   {
      cout << "***Warning: createGaussian(...) only defined for odd pixel"  << endl;
      cout << "            dimensions.  Undefined behavior for even sizes." << endl;
   }

   float pxCtrCol = (float)(nPixelsCol/2 + ctrCol);   
   float pxCtrRow = (float)(nPixelsRow/2 + ctrRow);   
   float distCol, distRow, distColSqNorm, distRowSqNorm;
   float denom = 2*M_PI*sigmaCol*sigmaRow;
   for(int c=0; c<nPixelsCol; c++)
   {
      distCol = (float)c - pxCtrCol;
      distColSqNorm = distCol*distCol / (sigmaCol*sigmaCol);
      for(int r=0; r<nPixelsRow; r++)
      {
         distRow = (float)r - pxCtrRow;
         distRowSqNorm = distRow*distRow / (sigmaRow*sigmaRow);
         
         targPtr[c*nPixelsRow+r] = exp(-0.5*(distColSqNorm + distRowSqNorm)) / denom;
      }
   }
}


// Assume diameter^2 target memory has already been allocated
// This filter is used for edge detection.  Convolve with the
// kernel created by this function, and then look for the 
// zero-crossings
// As always, we expect an odd diameter
// For LoG kernels, we always assume square and symmetric,
// which is why there are no options for different dimensions
void createLaplacianOfGaussianKernel(float* targPtr,
                                     int    diameter)
{
   float pxCtr = (float)(diameter-1) / 2.0f;
   float dc, dr, dcSq, drSq;
   float sigma = diameter/10.0f;
   float sigmaSq = sigma*sigma;
   for(int c=0; c<diameter; c++)
   {
      dc = (float)c - pxCtr;
      dcSq = dc*dc;
      for(int r=0; r<diameter; r++)
      {
         dr = (float)r - pxCtr;
         drSq = dr*dr;
   
         float firstTerm  = (dcSq + drSq - 2*sigmaSq) / (sigmaSq * sigmaSq);
         float secondTerm = exp(-0.5 * (dcSq + drSq) / sigmaSq);
         targPtr[c*diameter+r] = firstTerm * secondTerm;
      }
   }
}

// Assume diameter^2 target memory has already been allocated
int createBinaryCircle(int* targPtr,
                       int  diameter)
{
   float pxCtr = (float)(diameter-1) / 2.0f;
   float rad;
   int seNonZero = 0;
   for(int c=0; c<diameter; c++)
   {
      for(int r=0; r<diameter; r++)
      {
         rad = sqrt((c-pxCtr)*(c-pxCtr) + (r-pxCtr)*(r-pxCtr));
         if(rad <= pxCtr+0.5)
         {
            targPtr[c*diameter+r] = 1;
            seNonZero++;
         }
         else
         {
            targPtr[c*diameter+r] = 0;
         }
      }
   }
   return seNonZero;
}

// Assume diameter^2 target memory has already been allocated
cudaImageHost createBinaryCircle(int diameter)
{
   cudaImageHost out(diameter, diameter);
   float pxCtr = (float)(diameter-1) / 2.0f;
   float rad;
   for(int c=0; c<diameter; c++)
   {
      for(int r=0; r<diameter; r++)
      {
         rad = sqrt((c-pxCtr)*(c-pxCtr) + (r-pxCtr)*(r-pxCtr));
         if(rad <= pxCtr+0.5)
            out(c,r) = 1.0f;
         else
            out(c,r) = 0.0f;
      }
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
// Simple Timing Calls
void cpuStartTimer(void)
{
   // GPU Timer Functions
   cpuTimerVariable = 0;
   cutCreateTimer( &cpuTimerVariable );
   cutStartTimer(   cpuTimerVariable );
}

////////////////////////////////////////////////////////////////////////////////
// Stopping also resets the timer
// returns milliseconds
float cpuStopTimer(void)
{
   cutStopTimer( cpuTimerVariable );
   float cpuTime = cutGetTimerValue(cpuTimerVariable);
   cutDeleteTimer( cpuTimerVariable );
   return cpuTime;
}

////////////////////////////////////////////////////////////////////////////////
// Timing Calls for GPU -- this only counts GPU clock cycles, which will be 
// more precise for measuring GFLOPS and xfer rates, but shorter than wall time
void gpuStartTimer(void)
{
   cudaEventCreate(&eventTimerStart);
   cudaEventCreate(&eventTimerStop);
   cudaEventRecord(eventTimerStart);
}

////////////////////////////////////////////////////////////////////////////////
// Stopping also resets the timer
float gpuStopTimer(void)
{
   cudaEventRecord(eventTimerStop);
   cudaEventSynchronize(eventTimerStop);
   float gpuTime;
   cudaEventElapsedTime(&gpuTime, eventTimerStart, eventTimerStop);
   return gpuTime;
}

////////////////////////////////////////////////////////////////////////////////
// Read/Write images from/to files
void ReadFile(string fn, int* targPtr, int nCols, int nRows)
{
   ifstream in(fn.c_str(), ios::in);
   // We work with col-row format, but files written in row-col, so switch loop
   for(int r=0; r<nRows; r++)
      for(int c=0; c<nCols; c++)
         in >> targPtr[c*nCols+r];
   in.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing file in space-separated format
void WriteFile(string fn, int* srcPtr, int nCols, int nRows)
{
   ofstream out(fn.c_str(), ios::out);
   // We work with col-row format, but files written in row-col, so switch loop
   for(int r=0; r<nRows; r++)
   {
      for(int c=0; c<nCols; c++)
      {
         out << srcPtr[c*nRows+r] << " ";
      }
      out << endl;
   }
   out.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing image to stdout
void PrintArray(int* srcPtr, int nCols, int nRows)
{
   // We work with col-row format, but files written in row-col, so switch loop
   for(int r=0; r<nRows; r++)
   {
      cout << "\t";
      for(int c=0; c<nCols; c++)
      {
         cout << srcPtr[c*nRows+r] << " ";
      }
      cout << endl;
   }
}




////////////////////////////////////////////////////////////////////////////////
// Copy a 3D texture from a host (float*) array to a device cudaArray
// The extent should be specified with all dimensions in units of *elements*
void prepareCudaTexture(float* h_src, 
                        cudaArray *d_dst,
                        cudaExtent const texExtent)
{
   cudaMemcpy3DParms copyParams = {0};
   cudaPitchedPtr cppImgPsf = make_cudaPitchedPtr( (void*)h_src, 
                                                   texExtent.width*FLOAT_SZ,
                                                   texExtent.width,  
                                                   texExtent.height);
   copyParams.srcPtr   = cppImgPsf;
   copyParams.dstArray = d_dst;
   copyParams.extent   = texExtent;
   copyParams.kind     = cudaMemcpyHostToDevice;
   cutilSafeCall( cudaMemcpy3D(&copyParams) );
}
////////////////////////////////////////////////////////////////////////////////


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

////////////////////////////////////////////////////////////////////////////////
__global__ void  Mask_Union_Kernel( int* A, int* B, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;

   if( A[idx] + B[idx] > 0)
      devOut[idx] = 1;
   else
      devOut[idx] = 0;
}

////////////////////////////////////////////////////////////////////////////////
__global__ void  Mask_Intersect_Kernel( int* A, int* B, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   devOut[idx] = A[idx] * B[idx];
}

////////////////////////////////////////////////////////////////////////////////
// (A - B):   A is set to 0 if B is 1, otherwise A is left alone
__global__ void  Mask_Subtract_Kernel( int* A, int* B, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   if( B[idx] == 0)
      devOut[idx] = A[idx];
   else 
      devOut[idx] = 0;
}

////////////////////////////////////////////////////////////////////////////////
// (A - B):   A is set to 0 if B is 1, otherwise A is left alone
__global__ void  Mask_Difference_Kernel( int* A, int* B, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   
   if(A[idx] == B[idx])
      devOut[idx] = 0; 
   else
      devOut[idx] = 1; 

   // Should test if the extra algebra ops are worth removing the if-statement
   // Convert to {-1, +1}
   //int aval = A[idx]*2 - 1;
   //int bval = B[idx]*2 - 1;
   //devOut[idx] = (aval*bval+1)/2;
}

////////////////////////////////////////////////////////////////////////////////
__global__ void  Mask_Invert_Kernel( int* A, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   devOut[idx] = 1 - A[idx];
}


////////////////////////////////////////////////////////////////////////////////
// TODO: This is a very dumb/slow equal operator, actually won't even work
//       Perhaps have the threads atomicAdd to a globalMem location if !=
//__global__ void  Mask_CountDiff_Kernel( int* A, int* B, int* globalMemCount)
//{  
   //const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   //if(A[idx] != B[idx])
      //atomicAdd(numNotEqual, 1);
//}


////////////////////////////////////////////////////////////////////////////////
// TODO: Need to use reduction for this, but that can be kind of complicated
//       This operation destroys the input data, and the final result will be
//       stored in A[0]
__global__ void  Mask_Sum_Kernel( int* A, int valCount, int* scalarOut)
{  
   
   const int localIdx    = threadIdx.x;
   const int globalIdx   = blockDim.x*blockIdx.x + threadIdx.x;
   const int blockIdxOut = blockIdx.x / blockDim.x;

   while(valCount > 1)
   {
      int localCount = blockDim.x;
      while(localCount > 1)
      {
         localCount = localCount / 2;  
         if(localIdx < localCount)
            A[globalIdx] += A[globalIdx + localCount];
      }
   
      if(localIdx == 0)
         A[blockIdxOut] = A[globalIdx];

      valCount = valCount / blockDim.x;
   }

   if(globalIdx==0)
      scalarOut[0] = A[0];
}


////////////////////////////////////////////////////////////////////////////////
//
// This function takes an array of size N, and returns an array of size N/512
// that has the same sum as the original.  This method will need to be called
// recursively until the final size is one element that can be passed back to
// the host.
// 
// This kernel is not scalable.  I just assume that the block size will be 
// (256,1,1), so make sure you call it with that.  I did this to improve
// simplicity and speed slightly, at the expense of scalability

__global__ void  Image_SumReduceStep_Kernel( int* devBufIn,
                                             int* devBufOut,
                                             int  lastBlockSize)
{  
   // ONLY USE THIS FUNCTION WITH BLOCK SIZE = (256,1,1);
   // We write it for that to 
   __shared__ int sharedMem[4096];
   int* shmBuf1 = (int*)sharedMem;
   int* shmBuf2 = (int*)&sharedMem[512];

   int globalIdx = 512 * blockIdx.x + threadIdx.x;
   int localIdx  = threadIdx.x;

   shmBuf1[localIdx]     = 0;
   shmBuf1[localIdx+256] = 0;
   shmBuf2[localIdx]     = 0;
   shmBuf2[localIdx+256] = 0;

   if(blockIdx.x == gridDim.x-1)
   {
      if(localIdx+256 >= lastBlockSize) devBufIn[globalIdx+256] = 0;
      if(localIdx     >= lastBlockSize) devBufIn[globalIdx]     = 0;
   }

   // Now we reduce each block of 512 values (256 threads) to a single number

   shmBuf1[localIdx] = devBufIn[globalIdx] + devBufIn[globalIdx + 256]; __syncthreads();
   if(localIdx < 128) shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+128]; __syncthreads();
   if(localIdx < 64)  shmBuf1[localIdx] = shmBuf2[localIdx]+shmBuf2[localIdx+64];  __syncthreads();
   if(localIdx < 32)  shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+32];  __syncthreads();
   if(localIdx < 16)  shmBuf1[localIdx] = shmBuf2[localIdx]+shmBuf2[localIdx+16];  __syncthreads();
   if(localIdx < 8)   shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+8];   __syncthreads();
   if(localIdx < 4)   shmBuf1[localIdx] = shmBuf2[localIdx]+shmBuf2[localIdx+4];   __syncthreads();
   if(localIdx < 2)   shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+2];   __syncthreads();

   // 2 -> 1
   if(localIdx < 1)
      devBufOut[blockIdx.x] = shmBuf2[localIdx] + shmBuf2[localIdx + 1];
   __syncthreads(); 

}


// Yes, you really do need to pass in 2 full-sized, EXTRA, buffers
int Image_Sum(int* devImgToSum, int* devTemp1, int* devTemp2, int arraySize)
{
   // Yes, it seems silly to use two temp buffers to sum up an image, but
   // my goal was to make the reduction-kernel simple with the log(n) order of
   // growth, but not necessarily space-efficient
   
   cudaMemcpy(devTemp1, devImgToSum, arraySize*sizeof(int), cudaMemcpyDeviceToDevice);
   int* buf1 = devTemp1;
   int* buf2 = devTemp2;
   int* bufTemp;

   // The reduction kernel geometry is hardcoded b/c I wanted the code to be 
   // simple, not necessarily scalable
   dim3 BLOCK(256,1,1);
   int nEltsLeft = arraySize;

   while(nEltsLeft > 1)
   {
      int nBlocks = (nEltsLeft-1)/512+1;
      int lastBlockSize = ((nEltsLeft - (nBlocks-1)*512 ) - 1) % 512 + 1;
      dim3 GRID(nBlocks, 1, 1);

      Image_SumReduceStep_Kernel<<<GRID,BLOCK>>>(buf1, buf2, lastBlockSize);

      bufTemp = buf1; 
      buf1    = buf2;
      buf2    = bufTemp;

      nEltsLeft = nBlocks;

      cudaThreadSynchronize();
   }

   // Seems silly to do a memcpy like this to get one number out of the device
   // but I'm not aware of any other way (there probably is)
   int output; 
   cudaMemcpy(&output, buf1, sizeof(int), cudaMemcpyDeviceToHost);
   return output;
}






