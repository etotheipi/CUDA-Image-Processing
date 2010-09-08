#ifndef _IMAGE_WORKBENCH_H_CU_
#define _IMAGE_WORKBENCH_H_CU_

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"
#include "cudaMorphology.h.cu"
#include "cudaStructElt.h.cu"
#include "cudaImageHost.h"
#include "cudaImageDevice.h.cu"

////////////////////////////////////////////////////////////////////////////////
// This macro creates member method wrappers for each of the kernels created
// with the CREATE_3X3_MORPH_KERNEL macro.
//
// NOTE:  CREATE_3X3_MORPH_KERNEL macro creates KERNEL functions, this macro
//        creates member methods in ImageWorkbench that wrap those kernel
//        functions.  When calling these, you don't need to include the  
//        <<<GRID,BLOCK>>> as you would with a kernel function
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_WORKBENCH_3X3_FUNCTION( name ) \
   void name( void )\
   {  \
      /* User can only access PRIMARY and EXTRA buffers, not TEMP*/ \
      Morph3x3_##name##_Kernel<<<GRID_,BLOCK_>>>( \
                        bufferPtrA_->getDataPtr(), \
                        bufferPtrB_->getDataPtr(), \
                        imgCols_,    \
                        imgRows_);   \
      flipBuffers(); \
   } \

////////////////////////////////////////////////////////////////////////////////
// 
// ImageWorkbench
// 
// An image workbench is used when you have a single image to which you want
// to apply a sequence of dozens, hundreds or thousands of operations.
//
// The workbench copies the input data to the device once at construction, 
// and then applies all the operations, only extracting the result from the
// device when "copyResultToHost" is called.
//
// The workbench uses two primary image buffers, which are used to as input and
// output buffers, flipping back and forth every operation.  This is so that
// we don't need to keep copying the output back to the input buffer after each
// operation.
// 
// There's also on-demand temporary buffers, which may be needed for more
// advanced morphological operations.  For instance, the pruning and thinning
// kernels only *locate* pixels that need to be removed.  So we have to apply
// the pruning/thinning SEs into a temp buffer, and then subtract that buffer
// from the input.  This is why we have devExtraBuffers_.
//
// Static Data:
//
//    The static list of structuring elements ensures that we don't have to 
//    keep copying them into device memory every time we want to use them, 
//    and so that the numNonZero values can be calculated and stored with them.  
//    Otherwise, we would need to recalculate it every time.
//
////////////////////////////////////////////////////////////////////////////////
class ImageWorkbench
{
private:

   // All buffers in a workbench are the same size
   unsigned int imgCols_;
   unsigned int imgRows_;
   unsigned int imgElts_;
   unsigned int imgBytes_;

   // All 2D kernel functions will be called with the same geometry
   dim3  GRID_;
   dim3  BLOCK_;

   // Image data will jump back and forth between buf 1 and 2, each operation
   cudaImageDevice buffer1_;
   cudaImageDevice buffer2_;
   bool buf1_in_buf2_out_;

   // These two pointers will switch after every operation
   cudaImageDevice* bufferPtrA_;
   cudaImageDevice* bufferPtrB_;

   // We need to be able to allocate extra buffers for user to utilize, and
   // temporary buffers for various batch operations to use
   vector<cudaImageDevice> extraBuffers_;
   vector<cudaImageDevice> tempBuffers_;

   // Keep a master list of SEs and non-zero counts
   static vector<cudaImageDevice> masterListSE_;
   static vector<int>             masterListSENZ_;


   // We need temp buffers for operations like thinning, pruning
   void createExtraBuffer(void);
   void deleteExtraBuffer(void);
   void createTempBuffer(void);
   void deleteTempBuffer(void);

   // This gets called after every operation to switch Input/Output buffers ptrs
   void flipBuffers(void);

   cudaImageDevice* TEMP_BUF(int n);
public:
   void Initialize(cudaImageHost const & hostImg);

   cudaImageDevice* BUF_A(void) const {return bufferPtrA_;}
   cudaImageDevice* BUF_B(void) const {return bufferPtrB_;}
   cudaImageDevice* EXTRA_BUF(int n);

   dim3 getGridSize(void)  const {return GRID_;}
   dim3 getBlockSize(void) const {return BLOCK_;}
   void setBlockSize(dim3 newSize);

   // Calculate the device mem used by all IWBs and SEs
   static int calculateDeviceMemUsage(bool printToStdout=true);
   
   // Forking is the really just the same as copying
   // TODO:  not implemented yet
   void forkWorkbench(ImageWorkbench & iwb) const;

   static int addStructElt(int* hostSE, int ncols, int nrows);
   static int addStructElt(cudaImageHost const & seHost);

   // Default Constructor
   ImageWorkbench();
   ImageWorkbench(cudaImageHost const & hostImg) { Initialize(hostImg); }
   void copyResultToHost(cudaImageHost & putResultHere);
   
   // The basic morphological operations (CPU wrappers for GPU kernels)
   // NOTE: all batch functions, such as open, close, thinsweep, etc
   // are written so that when the user calls them, buffers A and B are 
   // distinctly before-and-after versions of the operation.  The
   // alternative is that A and B only contain the states before and
   // after the last SUB-operation, and then the user has no clean
   // way to determine if the image changed
   void GenericMorphOp(int seIndex, int targSum);
   void HitOrMiss(int seIndex); 
   void Erode(int seIndex);
   void Dilate(int seIndex);
   void Median(int seIndex);
   void Open(int seIndex);
   void Close(int seIndex);
   void FindAndRemove(int seIndex);

   // CPU wrappers for the mask op kernel functions which we need frequently
   void Union(int* mask2);
   void Intersect(int* mask2);
   void Subtract(int* mask2);
   void Invert(void);
   //int  NumPixelsChanged(void);
   //int  SumMask(void);

   void CopyBuffer(int* dst);
   static void CopyBuffer(int* src, int* dst, int bytes);

   /////////////////////////////////////////////////////////////////////////////
   // Thinning is a sequence of 8 hit-or-miss operations which each find
   // pixels contributing to the blob width, and then removes them from
   // the original image.  Very similar to skeletonization
   void ThinningSweep(void);

   /////////////////////////////////////////////////////////////////////////////
   // Pruning uses a sequence of 8 hit-or-miss operations to remove "loose ends"
   // from a thinned/skeletonized image.  
   void PruningSweep(void);



   // The macro calls below create wrappers for the optimized 3x3 kernel fns
   //
   //    void NAME(void)
   //    {
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(&bufA, &bufB, ...);
   //       flipBuffers();
   //    }
   //    void ZNAME(int* src, int* dst)
   //    {
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(src, dst, ...);
   //    }
   //
   CREATE_WORKBENCH_3X3_FUNCTION( Dilate );
   CREATE_WORKBENCH_3X3_FUNCTION( Erode );
   CREATE_WORKBENCH_3X3_FUNCTION( Median );
   CREATE_WORKBENCH_3X3_FUNCTION( Dilate4connect );
   CREATE_WORKBENCH_3X3_FUNCTION( Erode4connect );
   CREATE_WORKBENCH_3X3_FUNCTION( Median4connect );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin1 );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin2 );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin3 );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin4 );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin5 );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin6 );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin7 );
   CREATE_WORKBENCH_3X3_FUNCTION( Thin8 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune1 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune2 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune3 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune4 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune5 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune6 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune7 );
   CREATE_WORKBENCH_3X3_FUNCTION( Prune8 );

private:
   // These operations are the same as above, but with custom src-dst
   // and they don't flip the buffers.  These are "unsafe" for the
   // user to use, since he can destroy the current buffer, but the
   // developer can use them in IWB to ensure that batch operations
   // leave buffers A and B in a states that can be compared directly
   void ZGenericMorphOp(int seIndex, int targSum, int* src, int* dst);
   void ZHitOrMiss(int seIndex, int* src, int* dst);
   void ZErode(int seIndex, int* src, int* dst);
   void ZDilate(int seIndex, int* src, int* dst);
   void ZMedian(int seIndex, int* src, int* dst);
   void ZOpen(int seIndex, int* src, int* dst, int useTempBuf=0);
   void ZClose(int seIndex, int* src, int* dst, int useTempBuf=0);
   void ZFindAndRemove(int seIndex, int* src, int* dst, int useTempBuf=0);

   // CPU wrappers for the mask op kernel functions which we need frequently
   void ZUnion(int* mask2, int* src, int* dst);
   void ZIntersect(int* mask2, int* src, int* dst);
   void ZSubtract(int* mask2, int* src, int* dst);
   void ZInvert(int* src, int* dst);

};


#endif
