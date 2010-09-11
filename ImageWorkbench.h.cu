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

#define A -1000
#define B -1001


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
#define CREATE_3X3_WORKBENCH_METHOD( name )   \
   public: \
   void name( int srcBuf,  int dstBuf )  \
   {  \
      /* User can only access PRIMARY and EXTRA buffers, not TEMP*/ \
      int* srcPtr = getBufPtrAny(srcBuf, false)->getDataPtr(); \
      int* dstPtr = getBufPtrAny(dstBuf, false)->getDataPtr(); \
      Morph3x3_##name##_Kernel<<<GRID_2D_,BLOCK_2D_>>>(  \
                        srcPtr,    \
                        dstPtr,    \
                        imgCols_,  \
                        imgRows_); \
   } \
   void name( ) \
   {  \
      name( A, B ); \
      flipBuffers(); \
   } \
   \
   private: \
   void Z##name( int srcBuf, int dstBuf ) \
   {  \
      /* ZFunctions can access TEMP buffers too */ \
      int* srcPtr = getBufPtrAny(srcBuf, true)->getDataPtr(); \
      int* dstPtr = getBufPtrAny(dstBuf, true)->getDataPtr(); \
      Morph3x3_##name##_Kernel<<<GRID_2D_,BLOCK_2D_>>>( \
                        srcPtr,    \
                        dstPtr,    \
                        imgCols_,  \
                        imgRows_); \
   } \

////////////////////////////////////////////////////////////////////////////////
//
// These macros wrap the UNARY OPERATOR mask operations
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_MASK_UNARY_OP_WORKBENCH_METHOD( name )   \
   public: \
   void name( int srcBuf, int dstBuf ) \
   {  \
      /* User can only access PRIMARY and EXTRA buffers, not TEMP*/ \
      int* srcPtr = getBufPtrAny(srcBuf, false)->getDataPtr(); \
      int* dstPtr = getBufPtrAny(dstBuf, false)->getDataPtr(); \
      Mask_##name##_Kernel<<<GRID_1D_,BLOCK_1D_>>>( srcPtr, dstPtr );  \
   } \
   void name( ) \
   {  \
      name( A, B ); \
      flipBuffers(); \
   } \
   \
   private: \
   void Z##name( int srcBuf, int dstBuf ) \
   {  \
      /* ZFunctions can access TEMP buffers too */ \
      int* srcPtr = getBufPtrAny(srcBuf, true)->getDataPtr(); \
      int* dstPtr = getBufPtrAny(dstBuf, true)->getDataPtr(); \
      Mask_##name##_Kernel<<<GRID_1D_,BLOCK_1D_>>>( srcPtr, dstPtr); \
   } \


////////////////////////////////////////////////////////////////////////////////
//
// These macros wrap the BINARY OPERATOR mask operations
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_MASK_BINARY_OP_WORKBENCH_METHOD( name )   \
   public: \
   void name( int src2Buf, int src1Buf, int dstBuf ) \
   {  \
      /* User can only access PRIMARY and EXTRA buffers, not TEMP*/ \
      int* src1Ptr = getBufPtrAny(src1Buf, false)->getDataPtr();  \
      int* src2Ptr = getBufPtrAny(src2Buf, false)->getDataPtr();  \
      int* dstPtr  = getBufPtrAny(dstBuf,  false)->getDataPtr();  \
      Mask_##name##_Kernel<<<GRID_1D_,BLOCK_1D_>>>( src1Ptr, src2Ptr, dstPtr );\
   } \
   \
   void name( int src2Buf )  \
   {  \
      name( A, src2Buf, B); \
      flipBuffers(); \
   } \
   \
   private: \
   void Z##name( int src2Buf, int src1Buf, int dstBuf ) \
   {  \
      /* ZFunctions can access any buffers, including TEMP */ \
      int* src1Ptr = getBufPtrAny(src1Buf, true)->getDataPtr();  \
      int* src2Ptr = getBufPtrAny(src2Buf, true)->getDataPtr();  \
      int* dstPtr  = getBufPtrAny(dstBuf,  true)->getDataPtr();  \
      Mask_##name##_Kernel<<<GRID_1D_,BLOCK_1D_>>>( src1Ptr, src2Ptr, dstPtr );\
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
   dim3  GRID_1D_;
   dim3  GRID_2D_;
   dim3  BLOCK_1D_;
   dim3  BLOCK_2D_;


   // Image data will jump back and forth between buf 1 and 2, each operation
   cudaImageDevice buffer1_;
   cudaImageDevice buffer2_;

   // These two pointers will switch after every operation
   cudaImageDevice* bufferPtrA_;
   cudaImageDevice* bufferPtrB_;

   // The user should be to be able to allocate extra buffers 
   vector<cudaImageDevice> extraBuffers_;

   // Temporary buffers will be needed to carry our internal operations
   vector<cudaImageDevice> tempBuffers_;

   // We need to have locking, though, so that nested internal operations
   // don't overwrite buffers that are in use by the calling method
   vector<bool> tempBuffersLockFlag_;
   int  getTempBuffer(void);
   void releaseTempBuffer(int bufIdx);

   // Keep a master list of SEs and non-zero counts
   static vector<cudaImageDevice> masterListSE_;
   static vector<int>             masterListSENZ_;


   // We need temp buffers for operations like thinning, pruning
   void createExtraBuffer(void);
   void deleteExtraBuffer(void);
   void createTempBuffer(void);
   void deleteTempBuffer(void);

   // This method can get any buffer, PRIMARY, EXTRA or TEMP
   cudaImageDevice* getBufPtrAny(int idx, bool allowTemp=true);

   // All operations that don't specify src/dst will call this at the end
   // It switches bufA and bufB so that the next operation will use the 
   // previous output as input, and vice versa
   void flipBuffers(void);

public:

   // Primary constructor
   void Initialize(cudaImageHost const & hostImg);
   ImageWorkbench();
   ImageWorkbench(cudaImageHost const & hostImg);

   // IWB maintains a static list of all SEs, and we access them by index
   static int addStructElt(int* hostSE, int ncols, int nrows);
   static int addStructElt(cudaImageHost const & seHost);

   static cudaImageDevice const * getStructEltPtr(int i) {return &masterListSE_[i];}

   // This method is used to push/pull data to/from external locations
   void copyBufferToHost  ( int buf, cudaImageHost   & hostOut) const;
   void copyBufferToDevice( int buf, cudaImageDevice & hostOut) const;
   void copyBufferToHost  ( cudaImageHost   & hostOut) const;
   void copyBufferToDevice( cudaImageDevice & hostOut) const;
   void copyHostToBuffer  ( cudaImageHost   const & hostIn, int buf=A);
   void copyDeviceToBuffer( cudaImageDevice const & hostIn, int buf=A);

   // GPU Kernel geometry
   void setBlockSize1D(int nthreads);
   void setBlockSize2D(int ncols, int nrows);

   dim3 getBlockSize1D(void) const {return BLOCK_1D_;}
   dim3 getBlockSize2D(void) const {return BLOCK_2D_;}
   dim3 getGridSize1D(void)  const {return GRID_1D_;}
   dim3 getGridSize2D(void)  const {return GRID_2D_;}

   // This function can be used to access buffers directly, to copy data in 
   // or out of the workbench.  User can only access PRIMARY and EXTRA buffers
   cudaImageDevice* getBufferPtr(int bufIdx);


   
   /////////////////////////////////////////////////////////////////////////////
   // Standard set of morphological operators
   // NOTE: all batch functions, such as open, close, thinsweep, etc
   // are written so that when the user calls them, buffers A and B are 
   // distinctly before-and-after versions of the operation.  The
   // alternative is that A and B only contain the states before and
   // after the last SUB-operation, and then the user has no way to 
   // determine if the image changed
   void GenericMorphOp(int seIndex, int targSum)
      { GenericMorphOp(seIndex, targSum, A, B); flipBuffers(); }
   void HitOrMiss(int seIndex)
      { HitOrMiss(seIndex, A, B); flipBuffers(); }
   void Erode(int seIndex)
      { Erode(seIndex, A, B); flipBuffers(); }
   void Dilate(int seIndex)
      { Dilate(seIndex, A, B); flipBuffers(); }
   void Median(int seIndex)
      { Median(seIndex, A, B); flipBuffers(); }
   void Open(int seIndex)
      { Open(seIndex, A, B); flipBuffers(); }
   void Close(int seIndex)
      { Close(seIndex, A, B); flipBuffers(); }
   void FindAndRemove(int seIndex)
      { FindAndRemove(seIndex, A, B); flipBuffers(); }

   /////////////////////////////////////////////////////////////////////////////
   // Same morphological operators, but with customized src, dst
   void GenericMorphOp( int seIndex, int targSum, int srcBuf, int dstBuf);
   void HitOrMiss(      int seIndex, int srcBuf, int dstBuf);
   void Erode(          int seIndex, int srcBuf, int dstBuf);
   void Dilate(         int seIndex, int srcBuf, int dstBuf);
   void Median(         int seIndex, int srcBuf, int dstBuf);
   void Open(           int seIndex, int srcBuf, int dstBuf);
   void Close(          int seIndex, int srcBuf, int dstBuf);
   void FindAndRemove(  int seIndex, int srcBuf, int dstBuf);
   void CopyBuffer(                  int dstBuf, int srcBuf);

   int  SumImage(int srcBuf=A);
   int  CountChanged(void);

   /////////////////////////////////////////////////////////////////////////////
   // Thinning is a sequence of 8 hit-or-miss operations which each find
   // pixels contributing to the blob width, and then removes them from
   // the original image.  Very similar to skeletonization
   void ThinningSweep(void);

   /////////////////////////////////////////////////////////////////////////////
   // Pruning uses a sequence of 8 hit-or-miss operations to remove "loose ends"
   // from a thinned/skeletonized image.  
   void PruningSweep(void);

 
   /////////////////////////////////////////////////////////////////////////////
   // These macro calls create wrappers for the optimized 3x3 kernel functions
   // Each one creates 3 workbench methods:
   //
   // public:
   //    void NAME(input, output)
   //    {
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(input, output, ...);
   //    }
   //
   //    void NAME(void)  
   //    {
   //       NAME(bufA, bufB);
   //       flipBuffers();
   //    }
   //
   // private:
   //    void ZNAME(int* src, int* dst)
   //    {
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(src, dst, ...);
   //    }
   //
   CREATE_3X3_WORKBENCH_METHOD( Dilate );
   CREATE_3X3_WORKBENCH_METHOD( Erode );
   CREATE_3X3_WORKBENCH_METHOD( Median );
   CREATE_3X3_WORKBENCH_METHOD( Dilate4connect );
   CREATE_3X3_WORKBENCH_METHOD( Erode4connect );
   CREATE_3X3_WORKBENCH_METHOD( Median4connect );
   CREATE_3X3_WORKBENCH_METHOD( Thin1 );
   CREATE_3X3_WORKBENCH_METHOD( Thin2 );
   CREATE_3X3_WORKBENCH_METHOD( Thin3 );
   CREATE_3X3_WORKBENCH_METHOD( Thin4 );
   CREATE_3X3_WORKBENCH_METHOD( Thin5 );
   CREATE_3X3_WORKBENCH_METHOD( Thin6 );
   CREATE_3X3_WORKBENCH_METHOD( Thin7 );
   CREATE_3X3_WORKBENCH_METHOD( Thin8 );
   CREATE_3X3_WORKBENCH_METHOD( Prune1 );
   CREATE_3X3_WORKBENCH_METHOD( Prune2 );
   CREATE_3X3_WORKBENCH_METHOD( Prune3 );
   CREATE_3X3_WORKBENCH_METHOD( Prune4 );
   CREATE_3X3_WORKBENCH_METHOD( Prune5 );
   CREATE_3X3_WORKBENCH_METHOD( Prune6 );
   CREATE_3X3_WORKBENCH_METHOD( Prune7 );
   CREATE_3X3_WORKBENCH_METHOD( Prune8 );

   CREATE_MASK_UNARY_OP_WORKBENCH_METHOD( Invert );

   // Order of arguments can be confusing for binary ops, use Subtract for example
   //
   //    Subtract( bufN )                  - subtract bufN from the input buffer
   //                                        and put result in the output buffer
   //    Subtract( bufN, input, output)    - with three arguments, the last two are
   //                                        always input and output so that we have
   //                                        output = input - bufN
   //
   CREATE_MASK_BINARY_OP_WORKBENCH_METHOD( Union );
   CREATE_MASK_BINARY_OP_WORKBENCH_METHOD( Intersect );
   CREATE_MASK_BINARY_OP_WORKBENCH_METHOD( Subtract );
   CREATE_MASK_BINARY_OP_WORKBENCH_METHOD( Different );


private:

   /////////////////////////////////////////////////////////////////////////////
   // ZFunctions are special in 3 ways:
   //    1)  They can access the temporary buffers
   //    2)  They don't flip the buffers afterwards
   //    3)  They always require a source and destination
   //
   // The buffer flipping is important, since we like to be able to
   // compare the state immediately before and after an operation,
   // even batch operations
   //
   // So we use ZFunctions for things like open, close, thinsweep, etc,
   // so that BufA and BufB can be compared and we won't just be comparing
   // the last sub-operation in the batch
   void ZGenericMorphOp( int seIndex, int targSum,  int srcBuf, int dstBuf);
   void ZHitOrMiss(      int seIndex, int srcBuf, int dstBuf);
   void ZErode(          int seIndex, int srcBuf, int dstBuf);
   void ZDilate(         int seIndex, int srcBuf, int dstBuf);
   void ZMedian(         int seIndex, int srcBuf, int dstBuf);
   void ZOpen(           int seIndex, int srcBuf, int dstBuf);
   void ZClose(          int seIndex, int srcBuf, int dstBuf);
   void ZFindAndRemove(  int seIndex, int srcBuf, int dstBuf);

   void ZCopyBuffer(int dstBuf, int srcBuf);
   int  ZSumImage(int bufIdx);


};


#endif
