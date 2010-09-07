#ifndef _CUDA_IMAGE_HOST_H_CU_
#define _CUDA_IMAGE_HOST_H_CU_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//
// A very simple class for creating, storing and deleting image in *HOST* RAM
//
////////////////////////////////////////////////////////////////////////////////
class cudaImageHost
{
private:
   int* imgData_;
   int  imgCols_;
   int  imgRows_;
   int  imgElts_;
   int  imgBytes_;

   void Allocate(int ncols, int nrows);
   void MemcpyIn(int* dataIn);
   void Deallocate(void);

public:

   void resize(int ncols, int nrows); 

   cudaImageHost();
   cudaImageHost(int  ncols, int nrows);
   cudaImageHost(int* data, int ncols, int nrows);
   cudaImageHost(string filename, int ncols, int nrows);
   cudaImageHost(cudaImageHost const & img2);
   ~cudaImageHost();

   void  operator=(cudaImageHost const & img2);
   bool  operator==(cudaImageHost const & img2);

   int   operator()(int c, int r) const { return imgData_[c*imgRows_+r];}
   int & operator()(int c, int r)       { return imgData_[c*imgRows_+r];}
   int   operator[](int e)        const { return imgData_[e];           }
   int & operator[](int e)              { return imgData_[e];           }

   void readFile(string filename, int ncols, int nrows);
   void writeFile(string filename) const;
   void printImage(void) const;

   int* getDataPtr(void)  {return imgData_;}
   int  numCols(void)     {return imgCols_;}
   int  numRows(void)     {return imgRows_;}
   int  numElts(void)     {return imgElts_;}
   int  numBytes(void)    {return imgBytes_;}
};

#endif
