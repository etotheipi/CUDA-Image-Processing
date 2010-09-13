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
               int   imgRows,    
               int   imgCols,    
               int*  sePtr,    
               int   seRowRad,
               int   seColRad,
               int   seTargSum);

////////////////////////////////////////////////////////////////////////////////
// Standard 3x3 erosions, dilations and median filtering
DECLARE_3X3_MORPH_KERNEL( Dilate )
DECLARE_3X3_MORPH_KERNEL( Erode )
DECLARE_3X3_MORPH_KERNEL( Median )
DECLARE_3X3_MORPH_KERNEL( Dilate4connect )
DECLARE_3X3_MORPH_KERNEL( Erode4connect )
DECLARE_3X3_MORPH_KERNEL( Median4connect )

////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for THINNING
DECLARE_3X3_MORPH_KERNEL( Thin1 );
DECLARE_3X3_MORPH_KERNEL( Thin2 );
DECLARE_3X3_MORPH_KERNEL( Thin3 );
DECLARE_3X3_MORPH_KERNEL( Thin4 );
DECLARE_3X3_MORPH_KERNEL( Thin5 );
DECLARE_3X3_MORPH_KERNEL( Thin6 );
DECLARE_3X3_MORPH_KERNEL( Thin7 );
DECLARE_3X3_MORPH_KERNEL( Thin8 );
        
////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for PRUNING
DECLARE_3X3_MORPH_KERNEL( Prune1 );
DECLARE_3X3_MORPH_KERNEL( Prune2 );
DECLARE_3X3_MORPH_KERNEL( Prune3 );
DECLARE_3X3_MORPH_KERNEL( Prune4 );
DECLARE_3X3_MORPH_KERNEL( Prune5 );
DECLARE_3X3_MORPH_KERNEL( Prune6 );
DECLARE_3X3_MORPH_KERNEL( Prune7 );
DECLARE_3X3_MORPH_KERNEL( Prune8 );

#endif
