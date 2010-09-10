#include "cudaImageDevice.h.cu"


// Initialize a master list that
list<cudaImageDevice*> cudaImageDevice::masterDevImageList_ = list<cudaImageDevice*>(0);
int cudaImageDevice::totalDevMemUsed_ = 0;


////////////////////////////////////////////////////////////////////////////////
void cudaImageDevice::Allocate(int ncols, int nrows)
{
   imgCols_ = ncols;
   imgRows_ = nrows;
   imgElts_ = imgCols_*imgRows_;
   imgBytes_ = imgElts_*sizeof(int);

   if(ncols == 0 || nrows == 0)
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
void cudaImageDevice::Deallocate(void)
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
   imgCols_ = imgRows_ = imgElts_ = imgBytes_ = 0;
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageDevice::resize(int ncols, int nrows)
{
   if( imgElts_ == ncols*nrows)
   {
      // imgElts_ and imgBytes_ is already correct, no need to realloc
      imgCols_ = ncols;
      imgRows_ = nrows;
   }
   else
   {
      Deallocate();
      Allocate(ncols, nrows);
   }
}

////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::~cudaImageDevice()
{
   Deallocate();
}

////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::cudaImageDevice() :
   imgData_(NULL), imgCols_(0), imgRows_(0), imgElts_(0), imgBytes_(0) { }

////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::cudaImageDevice(int ncols, int nrows) :
   imgData_(NULL), imgCols_(0), imgRows_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(ncols, nrows);
}


////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::cudaImageDevice(cudaImageHost const & hostImg) :
   imgData_(NULL), imgCols_(0), imgRows_(0), imgElts_(0), imgBytes_(0)
{
   copyFromHost(hostImg);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::cudaImageDevice(cudaImageDevice const & devImg) :
   imgData_(NULL), imgCols_(0), imgRows_(0), imgElts_(0), imgBytes_(0)
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
void cudaImageDevice::copyFromHost  (int* hostPtr, int ncols, int nrows)
{
   resize(ncols, nrows);
   cudaMemcpy(imgData_, hostPtr, imgBytes_, cudaMemcpyHostToDevice);
}

/////
void cudaImageDevice::copyFromHost  (cudaImageHost const & hostImg)
{
   copyFromHost(hostImg.getDataPtr(), hostImg.numCols(), hostImg.numRows());
}

/////
void cudaImageDevice::copyToHost(int* hostPtr) const
{
   cudaMemcpy(hostPtr, imgData_, imgBytes_, cudaMemcpyDeviceToHost);
}

/////
void cudaImageDevice::copyToHost(cudaImageHost & hostImg) const
{
   hostImg.resize(imgCols_, imgRows_);
   copyToHost(hostImg.getDataPtr());
}

////////////////////////////////////////////////////////////////////////////////
// DEVICE <---> DEVICE 
/////
void cudaImageDevice::copyFromDevice(int* devicePtr, int ncols, int nrows)
{
   resize(ncols, nrows);
   cudaMemcpy(imgData_, devicePtr, imgBytes_, cudaMemcpyDeviceToDevice);
}

/////
void cudaImageDevice::copyFromDevice(cudaImageDevice const & devImg)
{
   copyFromDevice(devImg.getDataPtr(), devImg.numCols(), devImg.numRows());
}

/////
void cudaImageDevice::copyToDevice(int* devPtr) const
{
   cudaMemcpy(devPtr, imgData_, imgBytes_, cudaMemcpyDeviceToDevice);
}

/////
void cudaImageDevice::copyToDevice(cudaImageDevice & devImg) const
{
   devImg.resize(imgCols_, imgRows_);
   copyToDevice(devImg.getDataPtr());
}




////////////////////////////////////////////////////////////////////////////////
int cudaImageDevice::calculateDeviceMemoryUsage(bool dispStdout)
{
   float sizeMB = 1024.0f * 1024.0f;
   int totalBytes = 0;
   int nimg = (int)masterDevImageList_.size();
   int ct = 0;


   if(dispStdout)
      printf("\tDevice memory contains _%d_ cudaImageDevice objects\n", nimg);

   list<cudaImageDevice*>::iterator it;
   for( it  = masterDevImageList_.begin();
        it != masterDevImageList_.end();
        it++)
   {
      int nbytes = (*it)->imgBytes_;
      int wholeMB = nbytes / sizeMB;
      int fracMB  = (int)(10000 * (float)(nbytes - wholeMB*sizeMB) / (float)sizeMB);
      if(dispStdout)
         printf("\t\tDevice Image %3d (ID=%03d):  %4d x %4d,   %4d.%04d MB\n", ct, (*it)->id_, (*it)->imgCols_, (*it)->imgRows_, wholeMB, fracMB);
                                   
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

void cudaImageDevice::printMask(char zero, char one)
{
   cudaImageHost hptr;
   copyToHost(hptr);
   hptr.printMask(zero, one);
}




