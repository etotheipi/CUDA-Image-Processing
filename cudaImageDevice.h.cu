#ifndef _CUDA_IMAGE_DEVICE_H_CU_
#define _CUDA_IMAGE_DEVICE_H_CU_

#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cutil_inline.h>
#include "cudaImageHost.h"

////////////////////////////////////////////////////////////////////////////////
//
// A very simple class for creating, storing and deleting image in *DEVICE* RAM
//
////////////////////////////////////////////////////////////////////////////////
class cudaImageDevice 
{
private:
   int* imgData_;
   int  imgCols_;
   int  imgRows_;
   int  imgElts_;
   int  imgBytes_;

   void Allocate(int ncols, int nrows);
   void Deallocate(void);
   void MemcpyIn(int* dataIn);
   void MemcpyOut(int* dataOut);

   // Keep a master list of device memory allocations
   static list<cudaImageDevice*> masterDevImageList_;
   list<cudaImageDevice*>::iterator trackingIter_;
   static int totalDevMemUsed_;

public:
   void resize(int ncols, int nrows);
   int id_;

   cudaImageDevice();
   cudaImageDevice(int ncols, int nrows);
   cudaImageDevice(cudaImageHost & hostImg);
   ~cudaImageDevice();

   void copyFromHost(cudaImageHost & hostImg);
   void sendToHost(cudaImageHost & hostImg);

   operator int*() { return imgData_;}

   int* getDataPtr(void)  {return imgData_;}
   int  numCols(void)  {return imgCols_;}
   int  numRows(void)  {return imgRows_;}
   int  numElts(void)  {return imgElts_;}

   static int calculateDeviceMemoryUsage(bool dispStdout=false);
};

#endif
