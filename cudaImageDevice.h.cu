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

   // Keep a master list of device memory allocations
   static list<cudaImageDevice*> masterDevImageList_;
   list<cudaImageDevice*>::iterator trackingIter_;
   static int totalDevMemUsed_;

public:
   void resize(int ncols, int nrows);
   int id_;

   // Basic constructors
   cudaImageDevice();
   cudaImageDevice(int ncols, int nrows);
   cudaImageDevice(cudaImageHost   const & hostImg);
   cudaImageDevice(cudaImageDevice const &  devImg);
   ~cudaImageDevice();

   // Copying memory Host<->Device
   void copyFromHost  (int* hostPtr, int ncols, int nrows);
   void copyFromHost  (cudaImageHost const & hostImg);
   void copyToHost    (int* hostPtr) const;
   void copyToHost    (cudaImageHost & hostImg) const;

   // Copying memory Device<->Device
   void copyFromDevice  (int* devicePtr, int ncols, int nrows);
   void copyFromDevice  (cudaImageDevice const & deviceImg);
   void copyToDevice    (int* devicePtr) const;
   void copyToDevice    (cudaImageDevice & deviceImg) const;

   
   // Should only be used for images of zeros and ones, print D's for "Device"
   void printMask(char zero='.', char one='D');
   void writeFile(string filename);

   // Implicit cast to int* for functions that require int*
   operator int*() { return imgData_;}
   static int calculateDeviceMemoryUsage(bool dispStdout=false);

   int* getDataPtr(void) const {return imgData_;}
   int  numCols(void)    const {return imgCols_;}
   int  numRows(void)    const {return imgRows_;}
   int  numElts(void)    const {return imgElts_;}
   int  numBytes(void)   const {return imgBytes_;}

};

#endif
