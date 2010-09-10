
#include <stdio.h>
#include <iostream>
#include "cudaImageHost.h"
#include "cudaImageDevice.h.cu"
#include "ImageWorkbench.h.cu"


/////////////////////////////////////////////////////////////////////////////
// The static data members in the IWB class
vector<cudaImageDevice> ImageWorkbench::masterListSE_(0);
vector<int>             ImageWorkbench::masterListSENZ_(0);



/////////////////////////////////////////////////////////////////////////////
// Add the SE to the master list, calculate non-zero count, and return index
int ImageWorkbench::addStructElt(int* seHostPtr, int ncols, int nrows)
{
   int newIndex = (int)masterListSE_.size();
   cudaImageDevice seDev;
   masterListSE_.push_back( seDev );
   masterListSE_[newIndex].copyFromHost(seHostPtr, ncols, nrows);
   
   int nonZeroCount = 0;
   for(int e=0; e<ncols*nrows; e++)
      if(seHostPtr[e] == 1 || seHostPtr[e] == -1)
         nonZeroCount++;

   masterListSENZ_.push_back(nonZeroCount);
   return newIndex;
}

/////////////////////////////////////////////////////////////////////////////
int ImageWorkbench::addStructElt(cudaImageHost const & seHost)
{
   return addStructElt(seHost.getDataPtr(), seHost.numCols(), seHost.numRows());
}


/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::setBlockSize1D(int nthreads)
{
   BLOCK_2D_ = dim3(nthreads, 1, 1);
   GRID_2D_  = dim3(imgElts_/BLOCK_1D_.x, 1, 1);
}
/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::setBlockSize2D(int ncols, int nrows)
{
   BLOCK_2D_ = dim3(ncols, nrows, 1);
   GRID_2D_  = dim3(imgCols_/BLOCK_2D_.x, imgRows_/BLOCK_2D_.y, 1);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::createExtraBuffer(void)
{
   int newIndex = (int)extraBuffers_.size();
   cudaImageDevice newBuf;
   extraBuffers_.push_back(newBuf);
   extraBuffers_[newIndex].resize(imgCols_, imgRows_);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::deleteExtraBuffer(void)
{
   extraBuffers_.pop_back();
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::createTempBuffer(void)
{
   int newIndex = (int)tempBuffers_.size();
   cudaImageDevice newBuf;
   tempBuffers_.push_back(newBuf);
   tempBuffers_[newIndex].resize(imgCols_, imgRows_);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::deleteTempBuffer(void)
{
   tempBuffers_.pop_back();
}

/////////////////////////////////////////////////////////////////////////////
ImageWorkbench::ImageWorkbench() : 
   imgCols_(0),
   imgRows_(0),
   imgElts_(0),
   imgBytes_(0),
   buffer1_(0,0),
   buffer2_(0,0)
{ 
   // No code needed here
}


ImageWorkbench::ImageWorkbench(cudaImageHost const & hostImg) :
   imgCols_(0),
   imgRows_(0),
   imgElts_(0),
   imgBytes_(0),
   buffer1_(0,0),
   buffer2_(0,0)
{
   Initialize(hostImg);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Initialize(cudaImageHost const & hostImg)
{

   imgCols_  = hostImg.numCols();
   imgRows_  = hostImg.numRows();
   imgElts_  = hostImg.numElts();
   imgBytes_ = hostImg.numBytes();

   // 256 threads is a great block size for all 2.0+ devices, since that 
   // would be 6 blocks/multiprocessor which is less than the max of 8,
   // and more than enough to hide latency (assuming SHMEM and #registers
   // are low enough to allow 6 blocks/MP).
   setBlockSize1D(256);

   // For 2D, 32x8 dramatically reduces bank conflicts, compared to 16x16
   setBlockSize2D(32, 8);

   extraBuffers_ = vector<cudaImageDevice>(0);
   tempBuffers_  = vector<cudaImageDevice>(0);

   buffer1_.copyFromHost(hostImg);
   buffer2_.resize(imgCols_, imgRows_);

   // BufferA is input for a morph op, BufferB is the target, then switch
   bufferPtrA_ = &buffer1_;
   bufferPtrB_ = &buffer2_;
   cout << "Buffer1/2 locations = " << &buffer1_ << " " << &buffer2_ << endl;
}


/////////////////////////////////////////////////////////////////////////////
// Copy the current state of the buffer to the host
void ImageWorkbench::copyResultToHost(cudaImageHost & hostImg) const
{
   bufferPtrA_->copyToHost(hostImg);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyResultToDevice(cudaImageDevice & hostImg) const
{
   bufferPtrA_->copyToDevice(hostImg);
}


/////////////////////////////////////////////////////////////////////////////
// This method is used to push/pull data to/from external locations
void ImageWorkbench::copyBufferToHost( int bufIdx,
                                       cudaImageHost & hostOut) const
{
   // Need to do it this way because using getBufferPtr() is not const and
   // I want this function to be const
   if(bufIdx == A)
      bufferPtrA_->copyToHost(hostOut);
   else if(bufIdx == B)
      bufferPtrB_->copyToHost(hostOut);
   else if(bufIdx > 0)
      extraBuffers_[bufIdx-1].copyToHost(hostOut);
   else
      cout << "***ERROR: user has no access to TEMP buffers" << endl;
}
/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyBufferToDevice( int bufIdx,
                                         cudaImageDevice & devOut) const
{
   if(bufIdx == A)
      bufferPtrA_->copyToDevice(devOut);
   else if(bufIdx == B)
      bufferPtrB_->copyToDevice(devOut);
   else if(bufIdx > 0)
      extraBuffers_[bufIdx-1].copyToDevice(devOut);
   else
      cout << "***ERROR: user has no access to TEMP buffers" << endl;
}
/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyHostToBuffer( cudaImageHost const & hostIn,
                                       int bufIdx)
{
   if(hostIn.numCols() == imgCols_ && hostIn.numRows() == imgRows_)
      getBufferPtr(bufIdx)->copyFromHost(hostIn);
   else
   {
      printf("***ERROR:  can only copy images of same size as workbench (%dx%d)",
                                    imgCols_, imgRows_);
   }
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyDeviceToBuffer( cudaImageDevice const & devIn,
                                         int bufIdx)
{
   if(devIn.numCols() == imgCols_ && devIn.numRows() == imgRows_)
      getBufferPtr(bufIdx)->copyFromDevice(devIn);
   else
   {
      printf("***ERROR:  can only copy images of same size as workbench (%dx%d)",
                                    imgCols_, imgRows_);
   }
}


/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::flipBuffers(void)
{
   if(bufferPtrA_ == &buffer2_)
   {
      bufferPtrA_ = &buffer1_;
      bufferPtrB_ = &buffer2_;
   }
   else
   {
      bufferPtrA_ = &buffer2_;
      bufferPtrB_ = &buffer1_;
   }
}

/////////////////////////////////////////////////////////////////////////////
cudaImageDevice* ImageWorkbench::getBufferPtr( int idx )
{
   return getBufPtrAny(idx, false);
}

/////////////////////////////////////////////////////////////////////////////
cudaImageDevice* ImageWorkbench::getBufPtrAny( int idx, bool allowTemp)
{
   cudaImageDevice* out = NULL;

   if(idx == A)
      out = bufferPtrA_;
   else if(idx == B)
      out = bufferPtrB_;
   else if(idx > 0)  // Extra buffers 1 to N
   {
      while(idx > (int)extraBuffers_.size())
         createExtraBuffer();
      out = &extraBuffers_[idx-1];
   }
   else if(idx < 0)  // Temporary buffers, -1 to -N
   {
      if(allowTemp)
      {
         while(idx+1 > (int)tempBuffers_.size())
            createTempBuffer();
         out = &tempBuffers_[idx];
      }
      else
         cout << "***ERROR:  temp buffers only accessible to IWB methods"<<endl;
   }
   else
      cout << "***ERROR:  no buffer index " << idx << " is invalid" << endl;
      
   return out;
}

////////////////////////////////////////////////////////////////////////////////
//
// Finally, we get to define all the morphological operators!
//
// These are CPU methods which wrap the GPU kernel functions
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::GenericMorphOp(int seIndex, int targSum, int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufferPtr(srcBuf)->getDataPtr(),
                  getBufferPtr(dstBuf)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  targSum);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::HitOrMiss(int seIndex, int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufferPtr(srcBuf)->getDataPtr(),
                  getBufferPtr(dstBuf)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,
                  se->numRows()/2,
                  masterListSENZ_[seIndex]);
}

////////////////////////////////////////////////////////////////////////////////
// In out implementation, HitOrMiss is identical to erosion.  Theoretically,
// the difference is that the erode operator is expecting an SE that consists
// only of 1s (ON) and 0s (DONTCARE), while the HitOrMiss operation takes
// SEs that also have -1s (OFF).  However, this implementation allows -1s in 
// any SE, so they are interchangeable.
void ImageWorkbench::Erode(int seIndex, int srcBuf, int dstBuf)
{
   HitOrMiss(seIndex, srcBuf, dstBuf);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Dilate(int seIndex, int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufferPtr(srcBuf)->getDataPtr(),
                  getBufferPtr(dstBuf)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,
                  se->numRows()/2,
                  -masterListSENZ_[seIndex]+1);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Median(int seIndex, int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufferPtr(srcBuf)->getDataPtr(),
                  getBufferPtr(dstBuf)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,
                  se->numRows()/2,
                  0);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Open(int seIndex, int srcBuf, int dstBuf)
{
   ZDilate( seIndex, srcBuf,    1  );
   ZErode ( seIndex,   1,    dstBuf);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Close(int seIndex, int srcBuf, int dstBuf)
{
   ZDilate( seIndex, srcBuf,    1  );
   ZErode ( seIndex,   1,    dstBuf);
}

void ImageWorkbench::FindAndRemove(int seIndex, int srcBuf, int dstBuf)
{
   ZHitOrMiss(seIndex, srcBuf, 1);
   ZSubtract( 1,  srcBuf,  dstBuf);
}




////////////////////////////////////////////////////////////////////////////////
// Z FUNCTIONS (PRIVATE)
////////////////////////////////////////////////////////////////////////////////
// These operations are the same as above, but with custom src-dst
// and they don't flip the buffers.  These are "unsafe" for the
// user to use, since he can lose the current buffer, but the
// developer can use them in IWB to ensure that batch operations
// leave buffers A and B in a compare-able state
// 
// Here's what happens if you use regular methods for batch methods
// (THE WRONG WAY)
//       void ThinningSweep(idx)
//       {
//          Thin1();
//          Thin2();
//          Thin3();
//          Thin4();
//          ...
//          Thin8();
//    }
//
// The user wants to know whether the mask has reached equilibrium and
// calls NumChanged(), expecting to see 0 if it is at equilibrium.  The 
// problem is that since we've been flipping buffers constantly, the 
// NumChanged() function only gives us the num changed from the Thin8() 
// operation.  In fact, doing it this way, it is impossible for the user
// to check with whether Thin1, Thin2, ..., etc changed anything.
//
// Remember that SRC and DST are both device memory pointers
// which is another reason these are private
////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZGenericMorphOp(int seIndex, int targSum, int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srcBuf, true)->getDataPtr(),
                  getBufPtrAny(dstBuf, true)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  targSum);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZHitOrMiss(int seIndex,  int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srcBuf, true)->getDataPtr(),
                  getBufPtrAny(dstBuf, true)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  masterListSENZ_[seIndex]);
}

////////////////////////////////////////////////////////////////////////////////
// In out implementation, HitOrMiss is identical to erosion.  Theoretically,
// the difference is that the erode operator is expecting an SE that consists
// only of 1s (ON) and 0s (DONTCARE), while the HitOrMiss operation takes
// SEs that also have -1s (OFF).  However, this implementation allows -1s in 
// any SE, so they are interchangeable.
void ImageWorkbench::ZErode(int seIndex, int srcBuf, int dstBuf)
{
   ZHitOrMiss(seIndex, srcBuf, dstBuf);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZDilate(int seIndex, int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srcBuf, true)->getDataPtr(),
                  getBufPtrAny(dstBuf, true)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  -masterListSENZ_[seIndex]+1);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZMedian(int seIndex, int srcBuf, int dstBuf)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srcBuf, true)->getDataPtr(),
                  getBufPtrAny(dstBuf, true)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  0);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZOpen(int seIndex, int srcBuf, int dstBuf)
{
   ZErode ( seIndex, srcBuf,   1   );
   ZDilate( seIndex,   1,    dstBuf);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZClose(int seIndex, int srcBuf, int dstBuf)
{
   ZDilate( seIndex, srcBuf,   1   );
   ZErode(  seIndex,   1,    dstBuf);
}


////////////////////////////////////////////////////////////////////////////////
// Can't remember what this is supposed to be called, but it's the process by 
// which you do a Hit-or-Miss operation which is expected to return a sparse
// mask that fully intersects the original image, and then subtract. 
void ImageWorkbench::ZFindAndRemove(int seIndex, int srcBuf, int dstBuf)
{
   ZHitOrMiss(seIndex, srcBuf, 1);
   ZSubtract(1, srcBuf, dstBuf);
}

//int ImageWorkbench::NumPixelsChanged()
//{
   //MaskCountDiff_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                               // *bufferPtrA_, 
                               // *bufferPtrB_, 
                               //&nChanged);
   // No flip
//}


//int ImageWorkbench::SumMask()
//{
   //MaskSum_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                               // *bufferPtrA_, 
                               // *bufferPtrB_, 
                               //&nChanged);
   // No flip
//}

/////////////////////////////////////////////////////////////////////////////
// With all ZOperations implemented, I can finally implement complex batch
// operations like Thinning
void ImageWorkbench::ThinningSweep(void)
{
   // 1  (A->temp1->B)
   ZThin1   ( A,  1);
   ZSubtract( 1,  A,  B);

   // 2  (B->temp1->B)
   ZThin2   ( B,  1);
   ZSubtract( 1,  B,  B);

   // 3  (B->temp1->B)
   ZThin3   ( B,  1);
   ZSubtract( 1,  B,  B);

   // 4  (B->temp1->B)
   ZThin4   ( B,  1);
   ZSubtract( 1,  B,  B);

   // 5  (B->temp1->B)
   ZThin5   ( B,  1);
   ZSubtract( 1,  B,  B);

   // 6  (B->temp1->B)
   ZThin6   ( B,  1);
   ZSubtract( 1,  B,  B);

   // 7  (B->temp1->B)
   ZThin7   ( B,  1);
   ZSubtract( 1,  B,  B);

   // 8  (B->temp1->B)
   ZThin8   ( B,  1);
   ZSubtract( 1,  B,  B);

   // And we're done
   flipBuffers();
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::PruningSweep(void)
{
   // 1  (A->temp1->B)
   ZPrune1  ( A,  1);
   ZSubtract( 1,  A,  B);

   // 2  (B->temp1->B)
   ZPrune2  ( B,  1);
   ZSubtract( 1,  B,  B);

   // 3  (B->temp1->B)
   ZPrune3  ( B,  1);
   ZSubtract( 1,  B,  B);

   // 4  (B->temp1->B)
   ZPrune4  ( B,  1);
   ZSubtract( 1,  B,  B);

   // 5  (B->temp1->B)
   ZPrune5  ( B,  1);
   ZSubtract( 1,  B,  B);

   // 6  (B->temp1->B)
   ZPrune6  ( B,  1);
   ZSubtract( 1,  B,  B);

   // 7  (B->temp1->B)
   ZPrune7  ( B,  1);
   ZSubtract( 1,  B,  B);

   // 8  (B->temp1->B)
   ZPrune8  ( B,  1);
   ZSubtract( 1,  B,  B);

   // And we're done
   flipBuffers();
}



