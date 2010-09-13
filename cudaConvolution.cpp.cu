
#include <stdio.h> 
#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"


using namespace std;

__global__ void   convolveBasic( 
               int*   devInPtr,    
               int*   devOutPtr,    
               int    imgRows,    
               int    imgCols,    
               int*   devPsfPtr,    
               int    psfRowRad,
               int    psfColRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfRowRad, psfColRad); 
   shmOutput[localIdx] = 0.0f;

   const int psfStride = psfColRad*2+1;   
   const int psfPixels = psfStride*(psfRowRad*2+1);   
   int* shmPsf = (int*)&shmOutput[ROUNDUP32(localPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(devPsfPtr, shmPsf, psfPixels); 

   PREPARE_PADDED_RECTANGLE(psfRowRad, psfColRad); 


   __syncthreads();   


   int accum = 0.0f; 
   for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
   {   
      for(int coff=-psfColRad; coff<=psfColRad; coff++)   
      {   
         int psfRow = psfRowRad - roff;   
         int psfCol = psfColRad - coff;   
         int psfIdx = IDX_1D(psfRow, psfCol, psfStride);   
         int psfVal = shmPsf[psfIdx];   

         int shmPRRow = padRectRow + roff;   
         int shmPRCol = padRectCol + coff;   
         int shmPRIdx = IDX_1D(shmPRRow, shmPRCol, padRectStride);   
         accum += psfVal * shmPadRect[shmPRIdx];   
      }   
   }   
   shmOutput[localIdx] = accum;  
   __syncthreads();   

   devOutPtr[globalIdx] = shmOutput[localIdx];  
}


__global__ void   convolveBilateral( 
               int*   devInPtr,    
               int*   devOutPtr,    
               int    imgRows,    
               int    imgCols,    
               int*   devPsfPtr,    
               int    psfRowRad,
               int    psfColRad,
               int*   devIntPtr,    
               int    intensRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfRowRad, psfColRad); 
   shmOutput[localIdx] = 0.0f;

   const int padRectIdx = IDX_1D(padRectRow, padRectCol, padRectStride);
   const int psfStride = psfColRad*2+1;   
   const int psfPixels = psfStride*(psfRowRad*2+1);   
   int* shmPsf  = (int*)&shmOutput[ROUNDUP32(localPixels)];   
   int* shmPsfI = (int*)&shmPsf[ROUNDUP32(psfPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(devPsfPtr, shmPsf,  psfPixels); 
   COPY_LIN_ARRAY_TO_SHMEM(devIntPtr, shmPsfI, 2*intensRad+1);

   PREPARE_PADDED_RECTANGLE(psfRowRad, psfColRad); 


   __syncthreads();   


   int accum = 0.0f; 
   int myVal = shmPadRect[padRectIdx];
   for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
   {   
      for(int coff=-psfColRad; coff<=psfColRad; coff++)   
      {   
         int psfRow = psfRowRad - roff;   
         int psfCol = psfColRad - coff;   
         int psfIdx = IDX_1D(psfRow, psfCol, psfStride);   
         int psfVal = shmPsf[psfIdx];   

         int shmPRRow = padRectRow + roff;   
         int shmPRCol = padRectCol + coff;   
         int shmPRIdx = IDX_1D(shmPRRow, shmPRCol, padRectStride);   
         int thatVal = shmPadRect[shmPRIdx];

         int intVal = shmPsfI[(int)(thatVal-myVal+intensRad)];

         accum += psfVal * intVal *shmPadRect[shmPRIdx];   
      }   
   }   
   shmOutput[localIdx] = accum;  
   __syncthreads();   

   devOutPtr[globalIdx] = shmOutput[localIdx];  
}

