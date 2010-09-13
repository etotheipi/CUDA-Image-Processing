#include <string.h>
#include "cudaImageHost.h"


////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::Allocate(int nRows, int nCols)
{
   imgRows_ = nRows;
   imgCols_ = nCols;
   imgElts_ = imgRows_*imgCols_;
   imgBytes_ = imgElts_*sizeof(int);

   if(nRows == 0 || nCols == 0)
      imgData_ = NULL;
   else
   {
      imgData_ = (int*)malloc(imgBytes_);
   }
}

void cudaImageHost::MemcpyIn(int* dataIn)
{
   memcpy(imgData_, dataIn, imgBytes_);
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::Deallocate(void)
{
   if(imgData_ != NULL)
      free(imgData_);
   imgData_ = NULL;
   imgRows_ = imgCols_ = imgElts_ = imgBytes_ = 0;
}

void cudaImageHost::resize(int nRows, int nCols)
{
   // If we already have the right amount of memory, don't do anything
   if( imgElts_ == nRows*nCols)
   {
      // imgElts_ and imgBytes_ already correct, don't need to realloc
      imgRows_ = nRows; 
      imgCols_ = nCols;
   }
   else
   {
      Deallocate();
      Allocate(nRows, nCols);
   }
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::~cudaImageHost()
{
   Deallocate();
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost() :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0) { }



////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(int nRows, int nCols) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(nRows, nCols);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(int* data, int nRows, int nCols) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(nRows, nCols);    
   MemcpyIn(data);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(string filename, int nRows, int nCols) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(nRows, nCols);
   readFile(filename, nRows, nCols);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(cudaImageHost const & i2) :
   imgData_(NULL), imgRows_(0), imgCols_(0), imgElts_(0), imgBytes_(0)
{
   Allocate(i2.imgRows_, i2.imgCols_);
   MemcpyIn(i2.imgData_);
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::operator=(cudaImageHost const & i2)
{
   resize(i2.imgRows_, i2.imgCols_);
   MemcpyIn(i2.imgData_);
}

////////////////////////////////////////////////////////////////////////////////
bool  cudaImageHost::operator==(cudaImageHost const & i2) const
{
   bool isEq = true;
   if(imgRows_ != i2.imgRows_ || imgCols_ != i2.imgCols_)
      isEq = false;
   else
      for(int e=0; e<imgElts_; e++)
         if(imgData_[e] != i2.imgData_[e])
            isEq = false;

   return isEq;
}

////////////////////////////////////////////////////////////////////////////////
// Most of CUDA stuff will be done in Row-major format, but files store data
// in Col-major format, which is why we switch the normal order of the loops
void cudaImageHost::readFile(string filename, int nRows, int nCols)
{
   resize(nRows, nCols);

   ifstream is(filename.c_str(), ios::in);
   for(int r=0; r<imgRows_; r++)
      for(int c=0; c<imgCols_; c++)
         is >> imgData_[r*imgCols_ + c];
   is.close();
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::writeFile(string filename) const
{
   ofstream os(filename.c_str(), ios::out);
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
         os << imgData_[r*imgCols_+c] << " ";
      os << endl;
   }
   os.close();
}


////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::printMask(char zero, char one) const
{
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
      {
         int val = imgData_[r*imgCols_+c];
         if(val == 0)
            cout << zero;
         else
            cout << one;
         cout << " ";
      }
      cout << endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::printImage(void) const
{
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
      {
         cout << imgData_[r*imgCols_+c] << endl;
      }
      cout << endl;
   }
}


// This is the dumbest, simplest algorithm I could come up with.  There are most
// definitely more efficient way to implement it.  I just want an to get an
// order-of-magnitude timing
void cudaImageHost::Dilate(cudaImageHost SE, cudaImageHost & target)
{
   int seH = SE.numRows();
   int seW = SE.numCols();

   int imgH = imgRows_;
   int imgW = imgCols_;

   target.resize(imgW, imgH);

}
