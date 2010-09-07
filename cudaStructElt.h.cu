#ifndef _CUDA_STRUCT_ELT_H_CU_
#define _CUDA_STRUCT_ELT_H_CU_

////////////////////////////////////////////////////////////////////////////////
// 
// Structuring Element
//
// Structuring elements (SE) are the Point-Spread Functions (PSF) of image 
// morphology.  We use {-1, 0, +1} for {OFF, DONTCARE, ON}
//
// NOTE:  A structuring element object is directly linked to the device memory
//        where the SE data resides.  This class allocates the device memory
//        on construction and frees it on destruction
// 
////////////////////////////////////////////////////////////////////////////////
class StructElt
{
private:
   int* devPtr_;
   int  seCols_;
   int  seRows_;
   int  seElts_;
   int  seBytes_;
   int  seNonZero_;

public:
   void init(int* hostSE, int nc, int nr)
   {
      int numNonZero = 0;
      for(int i=0; i<seElts_; i++)
         if(hostSE[i] == -1 || hostSE[i] == 1)
           numNonZero++;

      init(hostSE, nc, nr, numNonZero);
   }

   void init(int* hostSE, int nc, int nr, int senz)
   {
      seCols_ = nc;
      seRows_ = nr;
      seElts_ = seCols_ * seRows_;
      seBytes_ = seElts_ * INT_SZ;
      seNonZero_ = senz;
      cudaMalloc((void**)&devPtr_, seBytes_);
      cudaMemcpy(devPtr_, hostSE, seBytes_, cudaMemcpyHostToDevice);
   }

   StructElt() :
      devPtr_(NULL),
      seCols_(-1),
      seRows_(-1),
      seElts_(-1),
      seBytes_(-1),
      seNonZero_(0) {}

   StructElt(int* hostSE, int nc, int nr) { init(hostSE, nc, nr); }

   ~StructElt() { cudaFree(devPtr_ ); }

   int* getDevPtr(void)  const  {return devPtr_;}
   int  getCols(void)    const  {return seCols_;}
   int  getRows(void)    const  {return seRows_;}
   int  getElts(void)    const  {return seElts_;}
   int  getBytes(void)   const  {return seBytes_;}
   int  getNonZero(void) const  {return seNonZero_;}
};

#endif
