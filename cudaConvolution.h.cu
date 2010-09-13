#ifndef _CUDA_CONVOLUTION_H_CU_
#define _CUDA_CONVOLUTION_H_CU_


__global__ void   convolveBasic( 
               int*   imgInPtr,    
               int*   imgOutPtr,    
               int    imgRows,    
               int    imgCols,    
               int*   imgPsfPtr,    
               int    psfRowRad,
               int    psfColRad);


__global__ void   convolveBilateral( 
               int*   devInPtr,    
               int*   devOutPtr,    
               int    imgRows,    
               int    imgCols,    
               int*   devPsfPtr,    
               int    psfRowRad,
               int    psfColRad,
               int*   devPsfPtr,    
               int    intensRad);

#endif
