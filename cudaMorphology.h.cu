#ifndef _CUDA_MORPHOLOGY_H_CU_
#define _CUDA_MORPHOLOGY_H_CU_

#include <stdio.h>

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
               int   seTargSum);

#endif
