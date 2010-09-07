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
void cudaImageDevice::MemcpyIn(int* dataIn)
{
   if(imgBytes_ > 0)
      cudaMemcpy(imgData_, dataIn, imgBytes_, cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageDevice::MemcpyOut(int* dataOut)
{
   if(imgBytes_ > 0)
      cudaMemcpy(dataOut, imgData_, imgBytes_, cudaMemcpyDeviceToHost);
}


////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::~cudaImageDevice()
{
   Deallocate();
}

////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::cudaImageDevice() :
   imgData_(NULL),
   imgCols_(0),
   imgRows_(0),
   imgElts_(0),
   imgBytes_(0) { }

////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::cudaImageDevice(int ncols, int nrows)
{
   Allocate(ncols, nrows);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageDevice::cudaImageDevice(cudaImageHost & hostImg)
{
   Allocate(hostImg.numCols(), hostImg.numRows());
   MemcpyIn(hostImg.getDataPtr());
}


////////////////////////////////////////////////////////////////////////////////
void cudaImageDevice::resize(int ncols, int nrows)
{
   if( imgElts_ != ncols*nrows)
   {
      Deallocate();
      Allocate(ncols, nrows);
   }
   imgCols_ = ncols;
   imgRows_ = nrows;
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageDevice::copyFromHost(cudaImageHost & hostImg)
{
   resize( hostImg.numCols(), hostImg.numRows());
   MemcpyIn( hostImg.getDataPtr() );
}

////////////////////////////////////////////////////////////////////////////////
// We assume that the host image and device image are the same size
void cudaImageDevice::sendToHost(cudaImageHost & hostImg)
{
   hostImg.resize(imgCols_, imgRows_);
   MemcpyOut( hostImg.getDataPtr() );
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
      printf("\t\tTotal Dev Mem Used:                        %4d.%04d MB\n\n", wholeMB, fracMB);
   }
   
   return totalDevMemUsed_;
}





