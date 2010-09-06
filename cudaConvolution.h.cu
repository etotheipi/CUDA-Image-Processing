#ifndef _CUDA_CONVOLUTION_H_CU_
#define _CUDA_CONVOLUTION_H_CU_


__global__ void   convolveBasic( 
               int*   imgInPtr,    
               int*   imgOutPtr,    
               int    imgCols,    
               int    imgRows,    
               int*   imgPsfPtr,    
               int    psfColRad,
               int    psfRowRad);


__global__ void   convolveBilateral( 
               int*   devInPtr,    
               int*   devOutPtr,    
               int    imgCols,    
               int    imgRows,    
               int*   devPsfPtr,    
               int    psfColRad,
               int    psfRowRad,
               int*   devPsfPtr,    
               int    intensRad);

#endif
