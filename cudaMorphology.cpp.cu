#include <stdio.h>
#include <iostream>
#include "cudaConvUtilities.h.cu"
#include "cudaMorphology.h.cu"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// ***Generic Morphologoical Operation Kernel Function***
// 
//    This is the basis for *ALL* other morpohological operations.  Every 
//    morphological operation in this library can be traced back to this
//    (the optimized 3x3 ops are hardcoded/unrolled versions of this function)
//
//    For all morph operations, we use {-1, 0, +1} ~ {OFF, DONTCARE, ON}.
//    This mapping allows us to use direct integer multiplication and 
//    summing of SE and image components.  Integer multiplication is 
//    much faster than using lots of if-statements.
//
//    Erosion, dilation, median, and a variety of weird and unique 
//    morphological operations are created solely by adjusting the 
//    target sum argument (seTargSum).
// 
////////////////////////////////////////////////////////////////////////////////
//
// Target Sum Values:
//
// The following describes under what conditions the SE is considered to "hit"
// a chunk of the image, based on how many indvidual pixels it "hits":
//
//
//    Erosion:  Hit every non-zero pixel
//
//          If we hit every pixel, we get a +1 for every non-zero elt
//          Therefore, our target should be [seNonZero]
//
//    Dilation:  Hit at least one non-zero pixel
//
//          If we miss every single pixel:  sum == -seNonZero
//          If we hit one pixel:            sum == -seNonZero+2;
//          If we hit two pixels:           sum == -seNonZero+4;
//          ...
//          Therefore, our target should be [-seNonZero+1] or greater
//
//
//    Median:   More pixels hit than not hit
//       
//          Since each pixel-hit is a +1, and each pixel-miss is a -1,
//          the median is 1 if and only if there are more +1s than -1s.
//          Therefore, our target should be [0] or greater
//
//
// THEREFORE, consider the following structuring elt with 6 non-zero pixels:
// 
//         se =      0  1  0
//                   1  1  1
//                   1  1  0
//
//  Morph_Generic_Kernel(se, targSum =  6) ~  Erode
//  Morph_Generic_Kernel(se, targSum = -5) ~  Dilate
//  Morph_Generic_Kernel(se, targSum =  0) ~  Median
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void  Morph_Generic_Kernel( 
               int*  devInPtr,    
               int*  devOutPtr,    
               int   imgCols,    
               int   imgRows,    
               int*  sePtr,    
               int   seColRad,
               int   seRowRad,
               int   seTargSum)
{  

   CREATE_CONVOLUTION_VARIABLES(seColRad, seRowRad); 
   shmOutput[localIdx] = -1;

   const int seStride = seRowRad*2+1;   
   const int sePixels = seStride*(seColRad*2+1);   
   int* shmSE = (int*)&shmOutput[ROUNDUP32(localPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(sePtr, shmSE, sePixels); 

   PREPARE_PADDED_RECTANGLE_MORPH(seColRad, seRowRad); 


   __syncthreads();   

   int accumInt = 0;
   for(int coff=-seColRad; coff<=seColRad; coff++)   
   {   
      for(int roff=-seRowRad; roff<=seRowRad; roff++)   
      {   
         int seCol = seColRad + coff;   
         int seRow = seRowRad + roff;   
         int seIdx = IDX_1D(seCol, seRow, seStride);   
         int seVal = shmSE[seIdx];   

         int shmPRCol = padRectCol + coff;   
         int shmPRRow = padRectRow + roff;   
         int shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);   
         accumInt += seVal * shmPadRect[shmPRIdx];
      }   
   }   
   // If every pixel was identical as expected, accumInt==seTargSum
   if(accumInt >= seTargSum)
      shmOutput[localIdx] = 1;

   __syncthreads();   

   devOutPtr[globalIdx] = (shmOutput[localIdx] + 1) / 2;
}


////////////////////////////////////////////////////////////////////////////////
// Standard 3x3 erosions, dilations and median filtering

// Standard (8-connected) 3x3 morph operations
CREATE_3X3_MORPH_KERNEL( Dilate,      -8,  
                                             1,  1,  1,
                                             1,  1,  1,
                                             1,  1,  1);
CREATE_3X3_MORPH_KERNEL( Erode,        9,
                                             1,  1,  1,
                                             1,  1,  1,
                                             1,  1,  1);
CREATE_3X3_MORPH_KERNEL( Median,       0,
                                             1,  1,  1,
                                             1,  1,  1,
                                             1,  1,  1);
// 4-connected (cross-shaped) structuring elements
CREATE_3X3_MORPH_KERNEL( Dilate4connect, -4,  
                                             0,  1,  0,
                                             1,  1,  1,
                                             0,  1,  0);
CREATE_3X3_MORPH_KERNEL( Erode4connect,   5,
                                             0,  1,  0,
                                             1,  1,  1,
                                             0,  1,  0);
CREATE_3X3_MORPH_KERNEL( Median4connect,  0,
                                             0,  1,  0,
                                             1,  1,  1,
                                             0,  1,  0);



////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for THINNING
CREATE_3X3_MORPH_KERNEL( Thin1,         7,
                                             1,  1,  1,
                                             0,  1,  0,
                                            -1, -1, -1);
CREATE_3X3_MORPH_KERNEL( Thin2,         7,
                                            -1,  0,  1,
                                            -1,  1,  1,
                                            -1,  0,  1);
CREATE_3X3_MORPH_KERNEL( Thin3,         7,
                                            -1, -1, -1,
                                             0,  1,  0,
                                             1,  1,  1);
CREATE_3X3_MORPH_KERNEL( Thin4,         7,
                                             1,  0, -1,
                                             1,  1, -1,
                                             1,  0, -1);

CREATE_3X3_MORPH_KERNEL( Thin5,         6,
                                             0, -1, -1,
                                             1,  1, -1,
                                             0,  1,  0);
CREATE_3X3_MORPH_KERNEL( Thin6,         6,
                                             0,  1,  0,
                                             1,  1, -1,
                                             0, -1, -1);
CREATE_3X3_MORPH_KERNEL( Thin7,         6,
                                             0,  1,  0,
                                            -1,  1,  1,
                                            -1, -1,  0);
CREATE_3X3_MORPH_KERNEL( Thin8,         6,
                                            -1, -1,  0,
                                            -1,  1,  1,
                                             0,  1,  0);
        
////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for PRUNING
CREATE_3X3_MORPH_KERNEL( Prune1,         7,
                                             0,  1,  0,
                                            -1,  1, -1,
                                            -1, -1, -1);
CREATE_3X3_MORPH_KERNEL( Prune2,         7,
                                            -1, -1,  0,
                                            -1,  1,  1,
                                            -1, -1,  0);
CREATE_3X3_MORPH_KERNEL( Prune3,         7,
                                            -1, -1, -1,
                                            -1,  1, -1,
                                             0,  1,  0);
CREATE_3X3_MORPH_KERNEL( Prune4,         7,
                                             0, -1, -1,
                                             1,  1, -1,
                                             0, -1, -1);

CREATE_3X3_MORPH_KERNEL( Prune5,         7,
                                            -1, -1, -1,
                                             0,  1, -1,
                                             1,  0, -1);
CREATE_3X3_MORPH_KERNEL( Prune6,         7,
                                            -1, -1, -1,
                                            -1,  1,  0,
                                            -1,  0,  1);
CREATE_3X3_MORPH_KERNEL( Prune7,         7,
                                            -1,  0,  1,
                                            -1,  1,  0,
                                            -1, -1, -1);
CREATE_3X3_MORPH_KERNEL( Prune8,         7,
                                             1,  0, -1,
                                             0,  1, -1,
                                            -1, -1, -1);

