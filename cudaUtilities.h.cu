#ifndef _CUDA_UTILITIES_H_CU_
#define _CUDA_UTILITIES_H_CU_


#include <iostream>
#include <stdio.h>
#include <cutil_inline.h>
#include <stopwatch.h>


// MASSIVE SPEED HIT for using wrong integer-multiply for compute capability
#if __CUDA_ARCH__ == 100      // Device code path for compute capability 1.0

   #define IMULT(a, b)           (__mul24(a,b))
   #define IMULTADD(a, b, c)     (__mul24(a,b) + (c))

#elif __CUDA_ARCH__ == 200    // Device code path for compute capability 2.0

   #define IMULT(a, b)           ((a)*(b))
   #define IMULTADD(a, b, c)     ((a)*(b) + (c))

#elif !defined(__CUDA_ARCH__) // Host code path

   #define IMULT(a, b)           ((a)*(b))
   #define IMULTADD(a, b, c)     ((a)*(b) + (c))

#endif

#define IDX_1D(Row, Col, stride) (IMULTADD(Row, stride, Col))
#define ROW_2D(index, stride) (index / stride)
#define COL_2D(index, stride) (index % stride)
#define ROUNDUP32(integer) ( ((integer-1)/32 + 1) * 32 )

#define FLOAT_SZ sizeof(float)
#define INT_SZ   sizeof(int)

using namespace std;


////////////////////////////////////////////////////////////////////////////////
// Copy a 3D texture from a host (float*) array to a device cudaArray
// The extent should be specified with all dimensions in units of *elements*
inline void prepareCudaTexture(float* h_src, 
                        cudaArray *d_dst,
                        cudaExtent const texExtent);

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
void ReadFile(string fn, int* targPtr, int nRows, int nCols);

////////////////////////////////////////////////////////////////////////////////
// Writing file in space-separated format
void WriteFile(string fn, int* srcPtr, int nRows, int nCols);

////////////////////////////////////////////////////////////////////////////////
// Writing image to stdout
void PrintArray(int* srcPtr, int nRows, int nCols);

#endif
