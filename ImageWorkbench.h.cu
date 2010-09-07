#ifndef _IMAGE_WORKBENCH_H_CU_
#define _IMAGE_WORKBENCH_H_CU_

#include <stdio.h>
#include <iostream>
#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"
#include "cudaMorphology.h.cu"
#include "cudaStructElt.h.cu"
#include "cudaImageDevice.h.cu"
#include "cudaImageHost.h"


////////////////////////////////////////////////////////////////////////////////
// This macro simply creates the declarations for the above functions, to be
// used in the header file
////////////////////////////////////////////////////////////////////////////////
#define DECLARE_3X3_MORPH_KERNEL( name ) \
__global__ void  Morph3x3_##name##_Kernel(       \
               int*   devInPtr,          \
               int*   devOutPtr,          \
               int    imgCols,          \
               int    imgRows); 




////////////////////////////////////////////////////////////////////////////////
// This macro creates member method wrappers for each of the kernels created
// with the CREATE_3X3_MORPH_KERNEL macro.
//
// NOTE:  CREATE_3X3_MORPH_KERNEL macro creates KERNEL functions, this macro
//        creates member methods in MorphWorkbench that wrap those kernel
//        functions.  When calling these, you don't need to include the  
//        <<<GRID,BLOCK>>> as you would with a kernel function
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_MWB_3X3_FUNCTION( name ) \
   void name(void) \
   {  \
      Morph3x3_##name##_Kernel<<<GRID_,BLOCK_>>>( \
                        *devBufferPtrA_, \
                        *devBufferPtrB_, \
                        imageCols_,  \
                        imageRows_);  \
      flipBuffers(); \
   } \
\
   void Z##name(int* src, int* dst) \
   {  \
      Morph3x3_##name##_Kernel<<<GRID_,BLOCK_>>>( \
                        src, \
                        dst, \
                        imageCols_,  \
                        imageRows_);  \
   } 




////////////////////////////////////////////////////////////////////////////////
// 
// ImageWorkbench
// 
// A morphology workbench is used when you have a single image to which you want
// to apply a sequence of dozens, hundreds or thousands of mophology operations.
//
// The workbench copies the input data to the device once at construction, 
// and then applies all the operations, only extracting the result from the
// device when "fetchBuffer" is called.
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
//    masterSEList_:
//
//    This class keeps a master list of all structuring elements and all
//    workbenches.  The static list of structuring elements ensures that we
//    don't have to keep copying them into device memory every time we want 
//    to use them, and so that the numNonZero values can be stored and kept
//    with them.  Otherwise, we would need to recalculate it every time.
//
//    masterMwbList_:
//
//    Additionally, we keep a running list of pointers to every ImageWorkbench
//    ever created (set to null when destructor is called).  The only real
//    benefit of this is so that we can query how much device memory we are
//    using at any given time.  See the method, calculateDeviceMemUsage();
//
////////////////////////////////////////////////////////////////////////////////
class ImageWorkbench
{
private:

   // The locations of device memory that contain all of our stuff
   int* devBuffer1_;
   int* devBuffer2_;
   vector<int*> devExtraBuffers_;

   // We want to keep track of every MWB and structuring element created
   // so we can calculate the total memory usage of all workbenches, which 
   // would include all buffers and SEs
   static vector<ImageWorkbench*> masterMwbList_;
   static vector<StructElt>       masterSEList_;

   // This workbench should know where it is in the master MWB list
   int mwbID_;


   // These two pointers will switch after every operation
   int** devBufferPtrA_;
   int** devBufferPtrB_;

   // Keep pointers to the host memory, so we know where to get input
   // and where to put the result
   int* hostImageIn_;
   bool imageCopied_;
   
   // All buffers in a workbench are the same size:  the size of the image
   unsigned int imageCols_;
   unsigned int imageRows_;
   unsigned int imagePixels_;
   unsigned int imageBytes_;

   // All kernel functions will be called with the same geometry
   dim3  GRID_;
   dim3  BLOCK_;

   // We need temp buffers for operations like thinning, pruning
   void createExtraBuffer(void);
   void deleteExtraBuffer(void);
   int* getExtraBufferPtr(int bufIdx);

   // This gets called after every operation to switch Input/Output buffers ptrs
   void flipBuffers(void);

public:

   dim3 getGridSize(void)  const {return GRID_;}
   dim3 getBlockSize(void) const {return BLOCK_;}
   void setBlockSize(dim3 newSize);

   // Calculate the device mem used by all MWBs and SEs
   static int calculateDeviceMemUsage(bool printToStdout=true);
   
   // Forking is the really just the same as copying
   // TODO:  not implemented yet
   void forkWorkbench(ImageWorkbench & mwb) const;

   static int addStructElt(int* hostSE, int ncols, int nrows);

   // Default Constructor
   ImageWorkbench();

   // Constructor
   ImageWorkbench(int* imageStart, int cols, int rows, bool COPY=false);

   // Copy host data to device, and prepare kernel parameters
   void Initialize(int* imageStart, int cols, int rows, bool COPY=false);

   // Destructor
   ~ImageWorkbench();

   // Copy the current state of the buffer to the host
   void fetchResult(int* hostTarget) const;
   
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
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(&debBufA, &devBufB, ...);
   //       flipBuffers();
   //    }
   //    void ZNAME(int* src, int* dst)
   //    {
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(src, dst, ...);
   //    }
   //
   CREATE_MWB_3X3_FUNCTION( Dilate );
   CREATE_MWB_3X3_FUNCTION( Erode );
   CREATE_MWB_3X3_FUNCTION( Median );
   CREATE_MWB_3X3_FUNCTION( Dilate4connect );
   CREATE_MWB_3X3_FUNCTION( Erode4connect );
   CREATE_MWB_3X3_FUNCTION( Median4connect );
   CREATE_MWB_3X3_FUNCTION( Thin1 );
   CREATE_MWB_3X3_FUNCTION( Thin2 );
   CREATE_MWB_3X3_FUNCTION( Thin3 );
   CREATE_MWB_3X3_FUNCTION( Thin4 );
   CREATE_MWB_3X3_FUNCTION( Thin5 );
   CREATE_MWB_3X3_FUNCTION( Thin6 );
   CREATE_MWB_3X3_FUNCTION( Thin7 );
   CREATE_MWB_3X3_FUNCTION( Thin8 );
   CREATE_MWB_3X3_FUNCTION( Prune1 );
   CREATE_MWB_3X3_FUNCTION( Prune2 );
   CREATE_MWB_3X3_FUNCTION( Prune3 );
   CREATE_MWB_3X3_FUNCTION( Prune4 );
   CREATE_MWB_3X3_FUNCTION( Prune5 );
   CREATE_MWB_3X3_FUNCTION( Prune6 );
   CREATE_MWB_3X3_FUNCTION( Prune7 );
   CREATE_MWB_3X3_FUNCTION( Prune8 );

private:
   // These operations are the same as above, but with custom src-dst
   // and they don't flip the buffers.  These are "unsafe" for the
   // user to use, since he can destroy the current buffer, but the
   // developer can use them in MWB to ensure that batch operations
   // leave buffers A and B in a compare-able state
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
