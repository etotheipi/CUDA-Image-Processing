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
template<class DTYPE>
class cudaImageHost
{
private:
   DTYPE* imgData_;
   int  imgRows_;
   int  imgCols_;
   int  imgElts_;
   int  imgBytes_;

   void Allocate(int nRows, int nCols);
   void Deallocate(void);
   void MemcpyIn(DTYPE* dataIn);

public:

   void resize(int nRows, int nCols); 

   cudaImageHost();
   cudaImageHost(int  nRows, int nCols);
   cudaImageHost(DTYPE* data, int nRows, int nCols);
   cudaImageHost(string filename, int nRows, int nCols);
   cudaImageHost(cudaImageHost const & img2);
   ~cudaImageHost();

   void  operator=(cudaImageHost const & img2);
   bool  operator==(cudaImageHost const & img2) const;

   DTYPE   operator()(int r, int c) const { return imgData_[r*imgCols_+c];}
   DTYPE & operator()(int r, int c)       { return imgData_[r*imgCols_+c];}
   DTYPE   operator[](int e)        const { return imgData_[e];           }
   DTYPE & operator[](int e)              { return imgData_[e];           }

   void readFile(string filename, int nRows, int nCols);
   void writeFile(string filename) const;
   void printImage(void) const;
   void printMask(char zero='.', char one='H') const; // 'H' for "Host"

   DTYPE* getDataPtr(void) const {return imgData_;}
   int  numRows(void)      const {return imgRows_;}
   int  numCols(void)      const {return imgCols_;}
   int  numElts(void)      const {return imgElts_;}
   int  numBytes(void)     const {return imgBytes_;}

   // This method is really only for timing tests.  Obviously we created
   // this library so we can use the GPU for 50-200x speed up.
   void Dilate(cudaImageHost SE, cudaImageHost & target);
};



////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageHost<DTYPE>::Allocate(int nRows, int nCols)
{
   imgRows_ = nRows;
   imgCols_ = nCols;
   imgElts_ = imgRows_*imgCols_;
   imgBytes_ = imgElts_*sizeof(DTYPE);

   if(nRows == 0 || nCols == 0)
      imgData_ = NULL;
   else
   {
      imgData_ = (DTYPE*)malloc(imgBytes_);
   }
}

template<class DTYPE>
void cudaImageHost<DTYPE>::MemcpyIn(DTYPE* dataIn)
{
   memcpy(imgData_, dataIn, imgBytes_);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageHost<DTYPE>::Deallocate(void)
{
   if(imgData_ != NULL)
      free(imgData_);
   imgData_ = NULL;
   imgRows_ = imgCols_ = imgElts_ = imgBytes_ = 0;
}

template<class DTYPE>
void cudaImageHost<DTYPE>::resize(int nRows, int nCols)
{
   // If we already have the right amount of memory, don't do anything
   if( imgElts_ == nRows*nCols)
   {
      // imgElts_ and imgBytes_ already correct, don't need to realloc
      imgRows_ = nRows; 
      imgCols_ = nCols;
   }
   else
   {
      Deallocate();
      Allocate(nRows, nCols);
   }
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageHost<DTYPE>::~cudaImageHost()
{
   Deallocate();
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageHost<DTYPE>::cudaImageHost() :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0) { }



////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageHost<DTYPE>::cudaImageHost(int nRows, int nCols) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(nRows, nCols);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageHost<DTYPE>::cudaImageHost(DTYPE* data, int nRows, int nCols) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(nRows, nCols);    
   MemcpyIn(data);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageHost<DTYPE>::cudaImageHost(string filename, int nRows, int nCols) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(nRows, nCols);
   readFile(filename, nRows, nCols);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageHost<DTYPE>::cudaImageHost(cudaImageHost const & i2) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(i2.imgRows_, i2.imgCols_);
   MemcpyIn(i2.imgData_);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageHost<DTYPE>::operator=(cudaImageHost const & i2)
{
   resize(i2.imgRows_, i2.imgCols_);
   MemcpyIn(i2.imgData_);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
bool  cudaImageHost<DTYPE>::operator==(cudaImageHost const & i2) const
{
   bool isEq = true;
   if(imgRows_ != i2.imgRows_ || imgCols_ != i2.imgCols_)
      isEq = false;
   else
      for(int e=0; e<imgElts_; e++)
         if(imgData_[e] != i2.imgData_[e])
            isEq = false;

   return isEq;
}

////////////////////////////////////////////////////////////////////////////////
// Most of CUDA stuff will be done in Row-major format, but files store data
// in Col-major format, which is why we switch the normal order of the loops
template<class DTYPE>
void cudaImageHost<DTYPE>::readFile(string filename, int nRows, int nCols)
{
   resize(nRows, nCols);

   ifstream is(filename.c_str(), ios::in);
   for(int r=0; r<imgRows_; r++)
      for(int c=0; c<imgCols_; c++)
         is >> imgData_[r*imgCols_ + c];
   is.close();
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageHost<DTYPE>::writeFile(string filename) const
{
   ofstream os(filename.c_str(), ios::out);
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
         os << imgData_[r*imgCols_+c] << " ";
      os << endl;
   }
   os.close();
}


////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageHost<DTYPE>::printMask(char zero, char one) const
{
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
      {
         int val = imgData_[r*imgCols_+c];
         if(val == 0)
            cout << zero;
         else
            cout << one;
         cout << " ";
      }
      cout << endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageHost<DTYPE>::printImage(void) const
{
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
      {
         cout << imgData_[r*imgCols_+c] << endl;
      }
      cout << endl;
   }
}


// This is the dumbest, simplest algorithm I could come up with.  There are most
// definitely more efficient way to implement it.  I just want an to get an
// order-of-magnitude timing
template<class DTYPE>
void cudaImageHost<DTYPE>::Dilate(cudaImageHost SE, cudaImageHost & target)
{
   int seH = SE.numRows();
   int seW = SE.numCols();

   int imgH = imgRows_;
   int imgW = imgCols_;

   target.resize(imgW, imgH);

}

#endif
