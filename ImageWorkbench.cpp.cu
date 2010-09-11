
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
   BLOCK_1D_ = dim3(nthreads, 1, 1);
   GRID_1D_  = dim3(imgElts_/BLOCK_1D_.x, 1, 1);
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

   // Make sure there are as many flags as there are buffers
   tempBuffersLockFlag_.push_back(false);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::deleteTempBuffer(void)
{
   tempBuffers_.pop_back();
   tempBuffersLockFlag_.pop_back();
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

   // For 2D, 8x32 dramatically reduces bank conflicts, compared to 16x16
   setBlockSize2D(8, 32);

   /*
   cout << endl;
   cout << "***Initializing new ImageWorkbench object" << endl;
   printf("\tImage Size (numCols, numRows) == (%d, %d)\n", imgCols_, imgRows_);
   printf("\tEach buffer is %d bytes\n\n", imgBytes_);
   printf("\t1D block size is (%d, %d, %d)\n", BLOCK_1D_.x, BLOCK_1D_.y, BLOCK_1D_.z);
   printf("\t1D grid  size is (%d, %d, %d)\n", GRID_1D_.x,  GRID_1D_.y,  GRID_1D_.z);
   printf("\t2D block size is (%d, %d, %d)\n", BLOCK_2D_.x, BLOCK_2D_.y, BLOCK_2D_.z);
   printf("\t2D grid  size is (%d, %d, %d)\n", GRID_2D_.x,  GRID_2D_.y,  GRID_2D_.z);
   cout << endl;
   */

   extraBuffers_ = vector<cudaImageDevice>(0);
   tempBuffers_  = vector<cudaImageDevice>(0);
   tempBuffersLockFlag_  = vector<bool>(0);

   buffer1_.copyFromHost(hostImg);
   buffer2_.resize(imgCols_, imgRows_);

   // BufferA is input for a morph op, BufferB is the target, then switch
   bufferPtrA_ = &buffer1_;
   bufferPtrB_ = &buffer2_;
}



/////////////////////////////////////////////////////////////////////////////
// These methods are used to push/pull main buffer to/from external locations
void ImageWorkbench::copyBufferToHost(cudaImageHost & hostOut) const
{
   bufferPtrA_->copyToHost(hostOut);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::copyBufferToDevice(cudaImageDevice & devOut) const
{
   bufferPtrA_->copyToHost(devOut);
}

/////////////////////////////////////////////////////////////////////////////
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
int  ImageWorkbench::getTempBuffer(void)
{
   int bufIndexOut =  0;
   int vectIndex   = -1;
   do 
   {
      bufIndexOut--;
      vectIndex++;

      // Make sure that this buffer exists
      getBufPtrAny(bufIndexOut);

   } while(tempBuffersLockFlag_[vectIndex] == true);

   tempBuffersLockFlag_[vectIndex] = true;
   return bufIndexOut;
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::releaseTempBuffer(int bufIdx)
{
   int vectIndex = (-1)*bufIdx - 1;
   tempBuffersLockFlag_[vectIndex] = false;
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
         int vectIndex = (-1)*idx - 1;
         while(vectIndex+1 > (int)tempBuffers_.size())
            createTempBuffer();
         out = &tempBuffers_[vectIndex];
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
   int tmpBuf = getTempBuffer();
   ZErode ( seIndex, srcBuf, tmpBuf );
   ZDilate( seIndex, tmpBuf, dstBuf );
   releaseTempBuffer(tmpBuf);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Close(int seIndex, int srcBuf, int dstBuf)
{
   int tmpBuf = getTempBuffer();
   ZDilate( seIndex, srcBuf, tmpBuf );
   ZErode ( seIndex, tmpBuf, dstBuf );
   releaseTempBuffer(tmpBuf);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::FindAndRemove(int seIndex, int srcBuf, int dstBuf)
{
   int tmpBuf = getTempBuffer();
   ZHitOrMiss(seIndex, srcBuf, tmpBuf);
   ZSubtract(  tmpBuf, srcBuf, dstBuf);
   releaseTempBuffer(tmpBuf);
}


////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::CopyBuffer(int dstBuf, int srcBuf)
{ 
   cudaImageDevice* src = getBufferPtr(srcBuf);
   cudaImageDevice* dst = getBufferPtr(dstBuf);
   dst->copyFromDevice(*src);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZCopyBuffer(int dstBuf, int srcBuf=A)
{ 
   cudaImageDevice* src = getBufPtrAny(srcBuf);
   cudaImageDevice* dst = getBufPtrAny(dstBuf);
   dst->copyFromDevice(*src);
}

////////////////////////////////////////////////////////////////////////////////
int ImageWorkbench::SumImage(int srcBuf)
{
   return ZSumImage(srcBuf);
}

////////////////////////////////////////////////////////////////////////////////
int ImageWorkbench::ZSumImage(int bufIdx)
{
   // Yes, it seems silly to use two temp buffers to sum up an image, but
   // my goal was to make the reduction-kernel simple with the log(n) order of
   // growth, but not necessarily space-efficient

   // Also, if we are trying to sum a temp buffer here, we don't want to
   // overwrite when getting more temp buffers
   int buf1Idx = getTempBuffer();
   int buf2Idx = getTempBuffer();
   ZCopyBuffer(buf1Idx, bufIdx);  // copy (dst, src)
   int* buf1 = getBufPtrAny(buf1Idx)->getDataPtr();
   int* buf2 = getBufPtrAny(buf2Idx)->getDataPtr();
   int* bufTemp;

   // The reduction kernel geometry is hardcoded b/c I wanted the code to be 
   // simple, not necessarily scalable
   dim3 BLOCK(256,1,1);
   int nEltsLeft = imgElts_;

   while(nEltsLeft > 1)
   {
      int nBlocks = (nEltsLeft-1)/512+1;
      int lastBlockSize = ((nEltsLeft - (nBlocks-1)*512 ) - 1) % 512 + 1;
      dim3 GRID(nBlocks, 1, 1);

      Image_SumReduceStep_Kernel<<<GRID,BLOCK>>>(buf1, buf2, lastBlockSize);

      bufTemp = buf1; 
      buf1    = buf2;
      buf2    = bufTemp;

      nEltsLeft = nBlocks;

      cudaThreadSynchronize();
   }

   releaseTempBuffer(buf1Idx);
   releaseTempBuffer(buf2Idx);

   // Seems silly to do a memcpy like this to get one number out of the device
   // but I'm not aware of any other way (there probably is)
   int output; 
   cudaMemcpy(&output, buf1, sizeof(int), cudaMemcpyDeviceToHost);
   return output;
}


int ImageWorkbench::CountChanged(void)
{
   int tmpBuf = getTempBuffer();
   Different(A, B, tmpBuf);
   int sum = SumImage(tmpBuf);
   releaseTempBuffer(tmpBuf);
   return sum;
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
   int tmpBuf = getTempBuffer();
   ZErode ( seIndex, srcBuf, tmpBuf );
   ZDilate( seIndex, tmpBuf, dstBuf );
   releaseTempBuffer(tmpBuf);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZClose(int seIndex, int srcBuf, int dstBuf)
{
   int tmpBuf = getTempBuffer();
   ZDilate( seIndex, srcBuf, tmpBuf );
   ZErode(  seIndex, tmpBuf, dstBuf );
   releaseTempBuffer(tmpBuf);
}


////////////////////////////////////////////////////////////////////////////////
// Can't remember what this is supposed to be called, but it's the process by 
// which you do a Hit-or-Miss operation which is expected to return a sparse
// mask that fully intersects the original image, and then subtract. 
void ImageWorkbench::ZFindAndRemove(int seIndex, int srcBuf, int dstBuf)
{
   int tmpBuf = getTempBuffer();
   ZHitOrMiss( seIndex, srcBuf, tmpBuf );
   ZSubtract(  tmpBuf,  srcBuf, dstBuf );
   releaseTempBuffer(tmpBuf);
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
// I know what you're thinking:  why don't I use ZFindAndRemove to write
// the sweep functions.  Well, I would, except that these ops use the 
// opimized 3x3 kernels which cannot be invoked by FindAndRemove().  Oh well.
void ImageWorkbench::ThinningSweep(void)
{
   int T = getTempBuffer();

   // 1  (A->tmp->B)
   ZThin1   ( A,  T);
   ZSubtract( T,  A,  B);

   // 2  (B->tmp->B)
   ZThin2   ( B,  T);
   ZSubtract( T,  B,  B);

   // 3  (B->tmp->B)
   ZThin3   ( B,  T);
   ZSubtract( T,  B,  B);

   // 4  (B->tmp->B)
   ZThin4   ( B,  T);
   ZSubtract( T,  B,  B);

   // 5  (B->tmp->B)
   ZThin5   ( B,  T);
   ZSubtract( T,  B,  B);

   // 6  (B->tmp->B)
   ZThin6   ( B,  T);
   ZSubtract( T,  B,  B);

   // 7  (B->tmp->B)
   ZThin7   ( B,  T);
   ZSubtract( T,  B,  B);

   // 8  (B->tmp->B)
   ZThin8   ( B,  T);
   ZSubtract( T,  B,  B);

   // And we're done
   flipBuffers();

   releaseTempBuffer(T);
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::PruningSweep(void)
{
   int T = getTempBuffer();

   // 1  (A->tmp->B)
   ZPrune1  ( A,  T);
   ZSubtract( T,  A,  B);

   // 2  (B->tmp->B)
   ZPrune2  ( B,  T);
   ZSubtract( T,  B,  B);

   // 3  (B->tmp->B)
   ZPrune3  ( B,  T);
   ZSubtract( T,  B,  B);

   // 4  (B->tmp->B)
   ZPrune4  ( B,  T);
   ZSubtract( T,  B,  B);

   // 5  (B->tmp->B)
   ZPrune5  ( B,  T);
   ZSubtract( T,  B,  B);

   // 6  (B->tmp->B)
   ZPrune6  ( B,  T);
   ZSubtract( T,  B,  B);

   // 7  (B->tmp->B)
   ZPrune7  ( B,  T);
   ZSubtract( T,  B,  B);

   // 8  (B->tmp->B)
   ZPrune8  ( B,  T);
   ZSubtract( T,  B,  B);

   // And we're done
   flipBuffers();

   releaseTempBuffer(T);
}



