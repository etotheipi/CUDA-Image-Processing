
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
void ImageWorkbench::setBlockSize(dim3 newSize)
{
   BLOCK_ = newSize;
   GRID_ = dim3(imgCols_/BLOCK_.x, imgRows_/BLOCK_.y, 1);
}

/////////////////////////////////////////////////////////////////////////////
cudaImageDevice* ImageWorkbench::EXTRA_BUF(int n)
{
   while(n+1 > (int)extraBuffers_.size())
      createExtraBuffer();
   return &extraBuffers_[n];
}

/////////////////////////////////////////////////////////////////////////////
cudaImageDevice* ImageWorkbench::TEMP_BUF(int n)
{
   while(n+1 > (int)tempBuffers_.size())
      createTempBuffer();
   return &tempBuffers_[n];
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

   // 32x8 dramatically reduces bank conflicts, compared to 16x16
   int bx = 32;
   int by = 8;
   int gx = imgCols_/bx;
   int gy = imgRows_/by;
   BLOCK_ = dim3(bx, by, 1);
   GRID_  = dim3(gx, gy, 1);

   extraBuffers_ = vector<cudaImageDevice>(0);
   tempBuffers_  = vector<cudaImageDevice>(0);

   buffer1_.copyFromHost(hostImg);
   buffer2_.resize(imgCols_, imgRows_);

   // BufferA is input for a morph op, BufferB is the target, then switch
   bufferPtrA_ = &buffer1_;
   bufferPtrB_ = &buffer2_;
   buf1_in_buf2_out_ = true;

}


/////////////////////////////////////////////////////////////////////////////
// Copy the current state of the buffer to the host
void ImageWorkbench::copyResultToHost(cudaImageHost & hostImg)
{
   cudaThreadSynchronize();
   bufferPtrA_->copyToHost(hostImg);
}


/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::flipBuffers(void)
{
   if(buf1_in_buf2_out_ == false)
   {
      bufferPtrA_ = &buffer1_;
      bufferPtrB_ = &buffer2_;
      buf1_in_buf2_out_ = true;
   }
   else
   {
      bufferPtrA_ = &buffer2_;
      bufferPtrB_ = &buffer1_;
      buf1_in_buf2_out_ = false;
   }
}

/*

////////////////////////////////////////////////////////////////////////////////
//
// Finally, we get to define all the morphological operators!
//
// These are CPU methods which wrap the GPU kernel functions
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::GenericMorphOp(int seIndex, int targSum)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *bufferPtrA_,
                  *bufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  targSum);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::HitOrMiss(int seIndex)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *bufferPtrA_,
                  *bufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  se.getNonZero());
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
// In out implementation, HitOrMiss is identical to erosion.  Theoretically,
// the difference is that the erode operator is expecting an SE that consists
// only of 1s (ON) and 0s (DONTCARE), while the HitOrMiss operation takes
// SEs that also have -1s (OFF).  However, this implementation allows -1s in 
// any SE, so they are interchangeable.
void ImageWorkbench::Erode(int seIndex)
{
   HitOrMiss(seIndex);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Dilate(int seIndex)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *bufferPtrA_,
                  *bufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  -se.getNonZero()+1);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Median(int seIndex)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *bufferPtrA_,
                  *bufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  0);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Open(int seIndex)
{
   int* tbuf = getExtraBufferPtr(0);
   ZErode(seIndex, *bufferPtrA_, tbuf);
   ZDilate(seIndex, tbuf, *bufferPtrB_);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Close(int seIndex)
{
   int* tbuf = getExtraBufferPtr(0);
   ZDilate(seIndex, *bufferPtrA_, tbuf);
   ZErode(seIndex, tbuf, *bufferPtrB_);
   flipBuffers();
}

void ImageWorkbench::FindAndRemove(int seIndex)
{
   int* tbuf = getExtraBufferPtr(0);
   ZHitOrMiss(seIndex, *bufferPtrA_, tbuf);
   ZSubtract(tbuf, *bufferPtrA_, *bufferPtrB_);
   flipBuffers();
}


////////////////////////////////////////////////////////////////////////////////
//
// These are the basic binary mask operations
//
// These are CPU methods which wrap the GPU kernel functions
//
////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::Union(int* devMask2)
{
   MaskUnion_Kernel<<<GRID_,BLOCK_>>>(
                              *bufferPtrA_, 
                               devMask2, 
                              *bufferPtrB_);
   flipBuffers();
}

void ImageWorkbench::Intersect(int* devMask2)
{
   MaskIntersect_Kernel<<<GRID_,BLOCK_>>>(
                              *bufferPtrA_,
                               devMask2,
                              *bufferPtrB_);
   flipBuffers();
}

void ImageWorkbench::Subtract(int* devMask2)
{
   MaskSubtract_Kernel<<<GRID_,BLOCK_>>>(
                              *bufferPtrA_,
                               devMask2,
                              *bufferPtrB_);
   flipBuffers();
}

void ImageWorkbench::Invert()
{
   MaskInvert_Kernel<<<GRID_,BLOCK_>>>(
                              *bufferPtrA_, 
                              *bufferPtrB_);
   flipBuffers();
}

void ImageWorkbench::CopyBuffer(int* dst)
{
   cudaMemcpy(dst,   
              *bufferPtrA_,  
              imageBytes_, 
              cudaMemcpyDeviceToDevice);
}

// Since this is static, 
void ImageWorkbench::CopyBuffer(int* src, int* dst, int bytes)
{
   cudaMemcpy(dst,   
              src,
              bytes, 
              cudaMemcpyDeviceToDevice);
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
void ImageWorkbench::ZGenericMorphOp(int seIndex, int targSum, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  targSum);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZHitOrMiss(int seIndex, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  se.getNonZero());
}

////////////////////////////////////////////////////////////////////////////////
// In out implementation, HitOrMiss is identical to erosion.  Theoretically,
// the difference is that the erode operator is expecting an SE that consists
// only of 1s (ON) and 0s (DONTCARE), while the HitOrMiss operation takes
// SEs that also have -1s (OFF).  However, this implementation allows -1s in 
// any SE, so they are interchangeable.
void ImageWorkbench::ZErode(int seIndex, int* src, int* dst)
{
   //ZHitOrMiss(seIndex, int* src, int* dst);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZDilate(int seIndex, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  -se.getNonZero()+1);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZMedian(int seIndex, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  0);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZOpen(int seIndex, int* src, int* dst, int useTempBuf)
{
   int* tbuf = getExtraBufferPtr(useTempBuf);
   ZErode(seIndex, src, tbuf);
   ZDilate(seIndex, tbuf, dst);
}

////////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::ZClose(int seIndex, int* src, int* dst, int useTempBuf)
{
   int* tbuf = getExtraBufferPtr(useTempBuf);
   ZErode(seIndex, src, tbuf);
   ZDilate(seIndex, tbuf, dst);
}


////////////////////////////////////////////////////////////////////////////////
// Can't remember what this is supposed to be called, but it's the process by 
// which you do a Hit-or-Miss operation which is expected to return a sparse
// mask that fully intersects the original image, and then subtract. 
void ImageWorkbench::ZFindAndRemove(int seIndex, int* src, int* dst, int useTempBuf)
{
   int* tbuf = getExtraBufferPtr(useTempBuf);
   ZHitOrMiss(seIndex, src, tbuf);
   ZSubtract(tbuf, src, dst);
}

//int ImageWorkbench::NumPixelsChanged()
//{
   //MaskCountDiff_Kernel<<<GRID_,BLOCK_>>>(
                               // *bufferPtrA_, 
                               // *bufferPtrB_, 
                               //&nChanged);
   // No flip
//}


//int ImageWorkbench::SumMask()
//{
   //MaskSum_Kernel<<<GRID_,BLOCK_>>>(
                               // *bufferPtrA_, 
                               // *bufferPtrB_, 
                               //&nChanged);
   // No flip
//}

void ImageWorkbench::ZUnion(int* devMask2, int* src, int* dst)
{
   MaskUnion_Kernel<<<GRID_,BLOCK_>>>(
                               src, 
                               devMask2, 
                               dst);
}

void ImageWorkbench::ZIntersect(int* devMask2, int* src, int* dst)
{
   MaskIntersect_Kernel<<<GRID_,BLOCK_>>>(
                               src,
                               devMask2,
                               dst);
}

void ImageWorkbench::ZSubtract(int* devMask2, int* src, int* dst)
{
   MaskSubtract_Kernel<<<GRID_,BLOCK_>>>(
                               src,
                               devMask2,
                               dst);
}

void ImageWorkbench::ZInvert(int* src, int* dst)
{
   MaskInvert_Kernel<<<GRID_,BLOCK_>>>( src, dst);
}

/////////////////////////////////////////////////////////////////////////////
// With all ZOperations implemented, I can finally implement complex batch
// operations like Thinning
void ImageWorkbench::ThinningSweep(void)
{
   int* tbuf0 = getExtraBufferPtr(0);

   // 1  (A->B)
   ZThin1(*bufferPtrA_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrA_, *bufferPtrB_);

   // 2  (B->B)
   ZThin2(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 3  (B->B)
   ZThin3(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 4  (B->B)
   ZThin4(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 5  (B->B)
   ZThin5(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 6  (B->B)
   ZThin6(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 7  (B->B)
   ZThin7(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 8  (B->B)
   ZThin8(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // And we're done
   flipBuffers();
}

/////////////////////////////////////////////////////////////////////////////
void ImageWorkbench::PruningSweep(void)
{
   int* tbuf0 = getExtraBufferPtr(0);

   // 1  (A->B)
   ZPrune1(*bufferPtrA_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrA_, *bufferPtrB_);

   // 2  (B->B)
   ZPrune2(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 3  (B->B)
   ZPrune3(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 4  (B->B)
   ZPrune4(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 5  (B->B)
   ZPrune5(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 6  (B->B)
   ZPrune6(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 7  (B->B)
   ZPrune7(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // 8  (B->B)
   ZPrune8(*bufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *bufferPtrB_, *bufferPtrB_);

   // And we're done
   flipBuffers();
}
*/
