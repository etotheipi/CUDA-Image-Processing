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
   void Deallocate(void);
   void MemcpyIn(int* dataIn);

public:

   void resize(int ncols, int nrows); 

   cudaImageHost();
   cudaImageHost(int  ncols, int nrows);
   cudaImageHost(int* data, int ncols, int nrows);
   cudaImageHost(string filename, int ncols, int nrows);
   cudaImageHost(cudaImageHost const & img2);
   ~cudaImageHost();

   void  operator=(cudaImageHost const & img2);
   bool  operator==(cudaImageHost const & img2) const;

   int   operator()(int c, int r) const { return imgData_[c*imgRows_+r];}
   int & operator()(int c, int r)       { return imgData_[c*imgRows_+r];}
   int   operator[](int e)        const { return imgData_[e];           }
   int & operator[](int e)              { return imgData_[e];           }

   void readFile(string filename, int ncols, int nrows);
   void writeFile(string filename) const;
   void printImage(void) const;
   void printMask(char zero='.', char one='H') const; // 'H' for "Host"

   int* getDataPtr(void)  const {return imgData_;}
   int  numCols(void)     const {return imgCols_;}
   int  numRows(void)     const {return imgRows_;}
   int  numElts(void)     const {return imgElts_;}
   int  numBytes(void)    const {return imgBytes_;}
};

#endif
