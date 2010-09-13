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
   int  imgRows_;
   int  imgCols_;
   int  imgElts_;
   int  imgBytes_;

   void Allocate(int nRows, int nCols);
   void Deallocate(void);
   void MemcpyIn(int* dataIn);

public:

   void resize(int nRows, int nCols); 

   cudaImageHost();
   cudaImageHost(int  nRows, int nCols);
   cudaImageHost(int* data, int nRows, int nCols);
   cudaImageHost(string filename, int nRows, int nCols);
   cudaImageHost(cudaImageHost const & img2);
   ~cudaImageHost();

   void  operator=(cudaImageHost const & img2);
   bool  operator==(cudaImageHost const & img2) const;

   int   operator()(int r, int c) const { return imgData_[r*imgCols_+c];}
   int & operator()(int r, int c)       { return imgData_[r*imgCols_+c];}
   int   operator[](int e)        const { return imgData_[e];           }
   int & operator[](int e)              { return imgData_[e];           }

   void readFile(string filename, int nRows, int nCols);
   void writeFile(string filename) const;
   void printImage(void) const;
   void printMask(char zero='.', char one='H') const; // 'H' for "Host"

   int* getDataPtr(void)  const {return imgData_;}
   int  numRows(void)     const {return imgRows_;}
   int  numCols(void)     const {return imgCols_;}
   int  numElts(void)     const {return imgElts_;}
   int  numBytes(void)    const {return imgBytes_;}

   // This method is really only for timing tests.  Obviously we created
   // this library so we can use the GPU for 50-200x speed up.
   void Dilate(cudaImageHost SE, cudaImageHost & target);
};

#endif
