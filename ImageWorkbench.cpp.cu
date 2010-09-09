
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
   imgBytes_(0)
{ 
   // No code needed here
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
void ImageWorkbench::copyBufferToHost( BUF_TYPE bt,
                                       int idx, 
                                       cudaImageHost & hostOut) const
{
   // Need to do it this way because using getBufferPtr() is not const and
   // I want this function to be const
   if(bt == BUF_EXTRA)
      extraBuffers_[idx].copyToHost(hostOut);
   else
      if(idx==A)
         bufferPtrA_->copyToHost(hostOut);
      else
         bufferPtrB_->copyToHost(hostOut);
}
/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyBufferToDevice( BUF_TYPE bt,
                                         int idx,
                                         cudaImageDevice & devOut) const
{
   // Need to do it this way because using getBufferPtr() is not const and
   // I want this function to be const
   if(bt == BUF_EXTRA)
      extraBuffers_[idx].copyToDevice(devOut);
   else
      if(idx==A)
         bufferPtrA_->copyToDevice(devOut);
      else
         bufferPtrB_->copyToDevice(devOut);
}
/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyHostToBuffer( cudaImageHost const & hostIn,
                                       BUF_TYPE bt,
                                       int idx)
{
   if(hostIn.numCols() == imgCols_ && hostIn.numRows() == imgRows_)
      getBufferPtr(bt, idx)->copyFromHost(hostIn);
   else
   {
      printf("***ERROR:  can only copy images of same size as workbench (%dx%d)",
                                    imgCols_, imgRows_);
   }
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyDeviceToBuffer( cudaImageDevice const & devIn,
                                             BUF_TYPE bt,
                                             int idx)
{
   if(devIn.numCols() == imgCols_ && devIn.numRows() == imgRows_)
      getBufferPtr(bt, idx)->copyFromDevice(devIn);
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
cudaImageDevice* ImageWorkbench::getBufferPtr( BUF_TYPE type, 
                                               int idx)
{
   // BUF_TEMP not allowed
   return getBufPtrAny(type, idx, false);
}

/////////////////////////////////////////////////////////////////////////////
cudaImageDevice* ImageWorkbench::getBufPtrAny( BUF_TYPE type, 
                                               int idx, 
                                               bool allowTemp)
{
   if(type == BUF_PRIMARY)
   {
      if(idx == A)
         return bufferPtrA_;
      else if(idx == B)
         return bufferPtrA_;
      else
         cout << "***ERROR:  no primary buffer #" << idx << endl;
   }
   else if(type == BUF_EXTRA)
   {
      while(idx+1 > (int)extraBuffers_.size())
         createExtraBuffer();
      return &extraBuffers_[idx];
   }
   else if(type == BUF_TEMP)
   {
      if(allowTemp)
      {
         while(idx+1 > (int)tempBuffers_.size())
            createTempBuffer();
         return &tempBuffers_[idx];
      }
      else
         cout << "***ERROR:  temp buffers only accessible to IWB methods"<<endl;
   }
   else
      cout << "***ERROR:  no buffer-type " << type << endl;

   return NULL;
}

////////////////////////////////////////////////////////////////////////////////
//
// Finally, we get to define all the morphological operators!
//
// These are CPU methods which wrap the GPU kernel functions
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::GenericMorphOp(int seIndex, int targSum,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, false)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, false)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  targSum);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::HitOrMiss(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, false)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, false)->getDataPtr(),
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
void ImageWorkbench::Erode(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   HitOrMiss(seIndex, srctype, srcidx, dsttype, dstidx);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Dilate(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, false)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, false)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,
                  se->numRows()/2,
                  -masterListSENZ_[seIndex]+1);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Median(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, false)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, false)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,
                  se->numRows()/2,
                  0);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Open(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   ZDilate( seIndex, srctype, srcidx, BUF_TEMP, 0);
   ZErode ( seIndex, BUF_TEMP, 0, dsttype, dstidx);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Close(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   ZDilate( seIndex, srctype, srcidx, BUF_TEMP, 0);
   ZErode ( seIndex, BUF_TEMP, 0, dsttype, dstidx);
}

void ImageWorkbench::FindAndRemove(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   ZHitOrMiss(seIndex, srctype, srcidx, BUF_TEMP, 0);
   ZSubtract(srctype, srcidx, BUF_TEMP, 0, dsttype, dstidx);
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
void ImageWorkbench::ZGenericMorphOp(int seIndex, int targSum,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, true)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, true)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  targSum);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZHitOrMiss(int seIndex, 
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, true)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, true)->getDataPtr(),
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
void ImageWorkbench::ZErode(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   ZHitOrMiss(seIndex, srctype, srcidx, dsttype, dstidx);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZDilate(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, true)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, true)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  -masterListSENZ_[seIndex]+1);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZMedian(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   cudaImageDevice* se = &masterListSE_[seIndex];

   Morph_Generic_Kernel<<<GRID_2D_,BLOCK_2D_>>>(
                  getBufPtrAny(srctype, srcidx, true)->getDataPtr(),
                  getBufPtrAny(dsttype, dstidx, true)->getDataPtr(),
                  imgCols_,
                  imgRows_,
                  se->getDataPtr(),
                  se->numCols()/2,  // pass in radius, not diam (yeah, confusing)
                  se->numRows()/2,
                  0);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZOpen(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   ZErode( seIndex, srctype, srcidx, BUF_TEMP, 0);
   ZDilate(seIndex, BUF_TEMP, 0, dsttype, dstidx);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZClose(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   ZDilate( seIndex, srctype, srcidx, BUF_TEMP, 0);
   ZErode(  seIndex, BUF_TEMP, 0, dsttype, dstidx);
}


////////////////////////////////////////////////////////////////////////////////
// Can't remember what this is supposed to be called, but it's the process by 
// which you do a Hit-or-Miss operation which is expected to return a sparse
// mask that fully intersects the original image, and then subtract. 
void ImageWorkbench::ZFindAndRemove(int seIndex,
                              BUF_TYPE srctype, int srcidx,
                              BUF_TYPE dsttype, int dstidx)
{
   ZHitOrMiss(seIndex, srctype, srcidx, BUF_TEMP, 0);
   ZSubtract(BUF_TEMP, 0, srctype, srcidx, dsttype, dstidx);
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
   // 1  (A->temp0->B)
   ZThin1   ( BUF_PRIMARY, A,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, A,  BUF_PRIMARY, B);

   // 2  (B->temp0->B)
   ZThin2   ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 3  (B->temp0->B)
   ZThin3   ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 4  (B->temp0->B)
   ZThin4   ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 5  (B->temp0->B)
   ZThin5   ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 6  (B->temp0->B)
   ZThin6   ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 7  (B->temp0->B)
   ZThin7   ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 8  (B->temp0->B)
   ZThin8   ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // And we're done
   flipBuffers();
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::PruningSweep(void)
{
   // 1  (A->temp0->B)
   ZPrune1  ( BUF_PRIMARY, A,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, A,  BUF_PRIMARY, B);

   // 2  (B->temp0->B)
   ZPrune2  ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 3  (B->temp0->B)
   ZPrune3  ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 4  (B->temp0->B)
   ZPrune4  ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 5  (B->temp0->B)
   ZPrune5  ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 6  (B->temp0->B)
   ZPrune6  ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 7  (B->temp0->B)
   ZPrune7  ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // 8  (B->temp0->B)
   ZPrune8  ( BUF_PRIMARY, B,  BUF_TEMP,    0);
   ZSubtract( BUF_TEMP,    0,  BUF_PRIMARY, B,  BUF_PRIMARY, B);

   // And we're done
   flipBuffers();
}
