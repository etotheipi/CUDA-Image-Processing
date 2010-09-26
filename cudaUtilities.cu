#include <iostream>
#include <fstream>
#include "cudaUtilities.h.cu"

// Timer variables
unsigned int cpuTimerVariable;
cudaEvent_t eventTimerStart;
cudaEvent_t eventTimerStop;


////////////////////////////////////////////////////////////////////////////////
// Copy a 3D texture from a host (float*) array to a device cudaArray
// The extent should be specified with all dimensions in units of *elements*
inline void prepareCudaTexture(float* h_src, 
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
void ReadFile(string fn, int* targPtr, int nRows, int nCols)
{
   ifstream in(fn.c_str(), ios::in);
   // We work with Row-Col format, but files written in Col-Row, so switch loop
   for(int r=0; r<nRows; r++)
      for(int c=0; c<nCols; c++)
         in >> targPtr[r*nRows+c];
   in.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing file in space-separated format
void WriteFile(string fn, int* srcPtr, int nRows, int nCols)
{
   ofstream out(fn.c_str(), ios::out);
   // We work with Row-Col format, but files written in Col-Row, so switch loop
   for(int r=0; r<nRows; r++)
   {
      for(int c=0; c<nCols; c++)
      {
         out << srcPtr[r*nCols+c] << " ";
      }
      out << endl;
   }
   out.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing image to stdout
void PrintArray(int* srcPtr, int nRows, int nCols)
{
   // We work with Row-Col format, but files written in Col-Row, so switch loop
   for(int r=0; r<nRows; r++)
   {
      cout << "\t";
      for(int c=0; c<nCols; c++)
      {
         cout << srcPtr[r*nCols+c] << " ";
      }
      cout << endl;
   }
}
