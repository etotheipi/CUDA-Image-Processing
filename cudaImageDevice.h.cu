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
template<class DTYPE>
class cudaImageDevice 
{
private:
   DTYPE* imgData_;
   int  imgRows_;
   int  imgCols_;
   int  imgElts_;
   int  imgBytes_;

   void Allocate(int nRows, int nCols);
   void Deallocate(void);

   // Keep a master list of device memory allocations
   static list<cudaImageDevice*> masterDevImageList_;
   typename list<cudaImageDevice*>::iterator trackingIter_;
   static int totalDevMemUsed_;

public:
   void resize(int nRows, int nCols);
   int id_;

   // Basic constructors
   cudaImageDevice();
   cudaImageDevice(int nRows, int nCols);
   cudaImageDevice(cudaImageHost<DTYPE>   const & hostImg);
   cudaImageDevice(cudaImageDevice const &  devImg);
   ~cudaImageDevice();

   // Copying memory Host<->Device
   void copyFromHost  (DTYPE* hostPtr, int nRows, int nCols);
   void copyFromHost  (cudaImageHost<DTYPE> const & hostImg);
   void copyToHost    (DTYPE* hostPtr) const;
   void copyToHost    (cudaImageHost<DTYPE> & hostImg) const;

   // Copying memory Device<->Device
   void copyFromDevice  (DTYPE* devicePtr, int nRows, int nCols);
   void copyFromDevice  (cudaImageDevice const & deviceImg);
   void copyToDevice    (DTYPE* devicePtr) const;
   void copyToDevice    (cudaImageDevice & deviceImg) const;

   
   // Should only be used for images of zeros and ones, print D's for "Device"
   void printMask(char zero='.', char one='D');
   void writeFile(string filename);

   // Implicit cast to int* for functions that require int*
   operator DTYPE*() { return imgData_;}
   static int calculateDeviceMemoryUsage(bool dispStdout=false);

   DTYPE* getDataPtr(void) const {return imgData_;}
   int  numRows(void)      const {return imgRows_;}
   int  numCols(void)      const {return imgCols_;}
   int  numElts(void)      const {return imgElts_;}
   int  numBytes(void)     const {return imgBytes_;}

};

// Initialize a master list that
template<class DTYPE>
list<cudaImageDevice<DTYPE>*> cudaImageDevice<DTYPE>::masterDevImageList_ 
                                           = list<cudaImageDevice<DTYPE>*>(0);
template<class DTYPE>
int cudaImageDevice<DTYPE>::totalDevMemUsed_ = 0;


////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageDevice<DTYPE>::Allocate(int nRows, int nCols)
{
   imgRows_ = nRows;
   imgCols_ = nCols;
   imgElts_ = imgRows_*imgCols_;
   imgBytes_ = imgElts_*sizeof(DTYPE);

   if(nRows == 0 || nCols == 0)
      imgData_ = NULL;
   else
   {
      cudaMalloc((void**)&imgData_, imgBytes_);
      totalDevMemUsed_ += imgBytes_;

      masterDevImageList_.push_back(this);
      trackingIter_ = masterDevImageList_.end();  // this is one-past-the-end
      trackingIter_--;

      static int idVal = 99;
      idVal++;
      id_ = idVal;
   }
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageDevice<DTYPE>::Deallocate(void)
{
   if(imgData_ != NULL)
   {
      cudaFree(imgData_);
      totalDevMemUsed_ -= imgBytes_;
      if(trackingIter_ != masterDevImageList_.end())
      {
         masterDevImageList_.erase(trackingIter_);
         trackingIter_ = masterDevImageList_.end();
      }
   } 
   imgData_ = NULL;
   imgRows_ = imgCols_ = imgElts_ = imgBytes_ = 0;
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageDevice<DTYPE>::resize(int nRows, int nCols)
{
   if( imgElts_ == nRows*nCols)
   {
      // imgElts_ and imgBytes_ is already correct, no need to realloc
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
cudaImageDevice<DTYPE>::~cudaImageDevice()
{
   Deallocate();
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageDevice<DTYPE>::cudaImageDevice() :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0) { }

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageDevice<DTYPE>::cudaImageDevice(int nRows, int nCols) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(nRows, nCols);
}


////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageDevice<DTYPE>::cudaImageDevice(cudaImageHost<DTYPE> const & hostImg) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   copyFromHost(hostImg);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
cudaImageDevice<DTYPE>::cudaImageDevice(cudaImageDevice<DTYPE> const & devImg) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   copyFromDevice(devImg);
}




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// 
// MEMORY COPY WRAPPERS
//
// These 8 methods handle all the possible ways we might want to copy data in,
// out, or between device memory locations.  
//
// NOTE:  These methods are not designed to explicitly allocate anyone else's
//        memory, so if we have only a pointer to destination memory, we have 
//        to assume it is already allocated properly.
//
//        If we're passed a reference to a cudaImage, we will call resize()
//        before copying to it.
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// HOST <---> DEVICE 
/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyFromHost  (DTYPE* hostPtr, int nRows, int nCols)
{
   resize(nRows, nCols);
   cudaMemcpy(imgData_, hostPtr, imgBytes_, cudaMemcpyHostToDevice);
}

/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyFromHost  (cudaImageHost<DTYPE> const & hostImg)
{
   copyFromHost(hostImg.getDataPtr(), hostImg.numRows(), hostImg.numCols());
}

/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyToHost(DTYPE* hostPtr) const
{
   cudaMemcpy(hostPtr, imgData_, imgBytes_, cudaMemcpyDeviceToHost);
}

/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyToHost(cudaImageHost<DTYPE> & hostImg) const
{
   hostImg.resize(imgRows_, imgCols_);
   copyToHost(hostImg.getDataPtr());
}

////////////////////////////////////////////////////////////////////////////////
// DEVICE <---> DEVICE 
/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyFromDevice(DTYPE* devicePtr, int nRows, int nCols)
{
   resize(nRows, nCols);
   cudaMemcpy(imgData_, devicePtr, imgBytes_, cudaMemcpyDeviceToDevice);
}

/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyFromDevice(cudaImageDevice const & devImg)
{
   copyFromDevice(devImg.getDataPtr(), devImg.numRows(), devImg.numCols());
}

/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyToDevice(DTYPE* devPtr) const
{
   cudaMemcpy(devPtr, imgData_, imgBytes_, cudaMemcpyDeviceToDevice);
}

/////
template<class DTYPE>
void cudaImageDevice<DTYPE>::copyToDevice(cudaImageDevice & devImg) const
{
   devImg.resize(imgRows_, imgCols_);
   copyToDevice(devImg.getDataPtr());
}




////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
int cudaImageDevice<DTYPE>::calculateDeviceMemoryUsage(bool dispStdout)
{
   float sizeMB = 1024.0f * 1024.0f;
   int totalBytes = 0;
   int nimg = (int)masterDevImageList_.size();
   int ct = 0;


   if(dispStdout)
      printf("\tDevice memory contains _%d_ cudaImageDevice objects\n", nimg);

   typename list<cudaImageDevice*>::iterator it;
   for( it  = masterDevImageList_.begin();
        it != masterDevImageList_.end();
        it++)
   {
      int nbytes = (*it)->imgBytes_;
      int wholeMB = nbytes / sizeMB;
      int fracMB  = (int)(10000 * (float)(nbytes - wholeMB*sizeMB) / (float)sizeMB);
      if(dispStdout)
         printf("\t\tDevice Image %3d (ID=%03d):  %4d x %4d,   %4d.%04d MB\n", ct, (*it)->id_, (*it)->imgRows_, (*it)->imgCols_, wholeMB, fracMB);
                                   
      ct++;

      totalBytes += (*it)->imgBytes_;
   }

   if(dispStdout)
   {
      int wholeMB = totalBytes / sizeMB;
      int fracMB  = (int)(10000 * (float)(totalBytes - wholeMB*sizeMB) / (float)sizeMB);
      printf("\t\t-------------------------------------------------------\n");
      printf("\t\tTotal Device Memory Used:                  %4d.%04d MB\n\n", wholeMB, fracMB);
   }
   
   return totalDevMemUsed_;
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageDevice<DTYPE>::printMask(char zero, char one)
{
   cudaImageHost<DTYPE> hptr;
   copyToHost(hptr);
   hptr.printMask(zero, one);
}

////////////////////////////////////////////////////////////////////////////////
template<class DTYPE>
void cudaImageDevice<DTYPE>::writeFile(string filename)
{
   cudaImageHost<DTYPE> hptr;
   copyToHost(hptr);
   hptr.writeFile(filename);
}


#endif
