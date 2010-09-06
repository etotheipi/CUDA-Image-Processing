
#include <stdio.h> 
#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"


using namespace std;

__global__ void   convolveBasic( 
               int*   devInPtr,    
               int*   devOutPtr,    
               int    imgCols,    
               int    imgRows,    
               int*   devPsfPtr,    
               int    psfColRad,
               int    psfRowRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfColRad, psfRowRad); 
   shmOutput[localIdx] = 0.0f;

   const int psfStride = psfRowRad*2+1;   
   const int psfPixels = psfStride*(psfColRad*2+1);   
   int* shmPsf = (int*)&shmOutput[ROUNDUP32(localPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(devPsfPtr, shmPsf, psfPixels); 

   PREPARE_PADDED_RECTANGLE(psfColRad, psfRowRad); 


   __syncthreads();   


   int accum = 0.0f; 
   for(int coff=-psfColRad; coff<=psfColRad; coff++)   
   {   
      for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
      {   
         int psfCol = psfColRad - coff;   
         int psfRow = psfRowRad - roff;   
         int psfIdx = IDX_1D(psfCol, psfRow, psfStride);   
         int psfVal = shmPsf[psfIdx];   

         int shmPRCol = padRectCol + coff;   
         int shmPRRow = padRectRow + roff;   
         int shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);   
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
               int    imgCols,    
               int    imgRows,    
               int*   devPsfPtr,    
               int    psfColRad,
               int    psfRowRad,
               int*   devIntPtr,    
               int    intensRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfColRad, psfRowRad); 
   shmOutput[localIdx] = 0.0f;

   const int padRectIdx = IDX_1D(padRectCol, padRectRow, padRectStride);
   const int psfStride = psfRowRad*2+1;   
   const int psfPixels = psfStride*(psfColRad*2+1);   
   int* shmPsf  = (int*)&shmOutput[ROUNDUP32(localPixels)];   
   int* shmPsfI = (int*)&shmPsf[ROUNDUP32(psfPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(devPsfPtr, shmPsf,  psfPixels); 
   COPY_LIN_ARRAY_TO_SHMEM(devIntPtr, shmPsfI, 2*intensRad+1);

   PREPARE_PADDED_RECTANGLE(psfColRad, psfRowRad); 


   __syncthreads();   


   int accum = 0.0f; 
   int myVal = shmPadRect[padRectIdx];
   for(int coff=-psfColRad; coff<=psfColRad; coff++)   
   {   
      for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
      {   
         int psfCol = psfColRad - coff;   
         int psfRow = psfRowRad - roff;   
         int psfIdx = IDX_1D(psfCol, psfRow, psfStride);   
         int psfVal = shmPsf[psfIdx];   

         int shmPRCol = padRectCol + coff;   
         int shmPRRow = padRectRow + roff;   
         int shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);   
         int thatVal = shmPadRect[shmPRIdx];

         int intVal = shmPsfI[(int)(thatVal-myVal+intensRad)];

         accum += psfVal * intVal *shmPadRect[shmPRIdx];   
      }   
   }   
   shmOutput[localIdx] = accum;  
   __syncthreads();   

   devOutPtr[globalIdx] = shmOutput[localIdx];  
}

