#include "cudaImageHost.h"


////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::Allocate(int ncols, int nrows)
{
   //printf("***Allocate called!  ");
   imgCols_ = ncols;
   imgRows_ = nrows;
   imgElts_ = imgCols_*imgRows_;
   imgBytes_ = imgElts_*sizeof(int);

   if(ncols == 0 || nrows == 0)
      imgData_ = NULL;
   else
   {
      imgData_ = (int*)malloc(imgBytes_);
   }
   //printf("ID=%d\n", id_);
}

void cudaImageHost::MemcpyIn(int* dataIn)
{
   for(int e=0; e<imgElts_; e++)
      imgData_[e] = dataIn[e];
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::Deallocate(void)
{
   //printf("***Deallocate called!  ID=%d\n", id_); cout.flush();
   if(imgData_ != NULL)
   {
      //printImage(); cout.flush();
      free(imgData_);
   }
   imgData_ = NULL;
   imgCols_ = imgRows_ = imgElts_ = imgBytes_ = 0;
}

void cudaImageHost::resize(int ncols, int nrows)
{
   // If we already have the right amount of memory, don't do anything
   if( imgElts_ != ncols*nrows)
   {
      Deallocate();
      Allocate(ncols, nrows);
   }
   // We could have the same linear array size but different dimensions
   imgCols_ = ncols; 
   imgRows_ = nrows;
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::~cudaImageHost()
{
   Deallocate();
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost() :
   imgData_(NULL),
   imgCols_(0),
   imgRows_(0),
   imgElts_(0),
   imgBytes_(0) { }



////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(int ncols, int nrows)
{
   Allocate(ncols, nrows);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(int* data, int ncols, int nrows)
{
   Allocate(ncols, nrows);    
   MemcpyIn(data);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(string filename, int ncols, int nrows)
{
   Allocate(ncols, nrows);
   readFile(filename, ncols, nrows);
}

////////////////////////////////////////////////////////////////////////////////
cudaImageHost::cudaImageHost(cudaImageHost const & i2)
{
   Allocate(i2.imgCols_, i2.imgRows_);
   MemcpyIn(i2.imgData_);
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::operator=(cudaImageHost const & i2)
{
   resize(i2.imgCols_, i2.imgRows_);
   MemcpyIn(i2.imgData_);
}

////////////////////////////////////////////////////////////////////////////////
bool  cudaImageHost::operator==(cudaImageHost const & i2)
{
   bool isEq = true;
   if(imgCols_ != i2.imgCols_ || imgRows_ != i2.imgRows_)
      isEq = false;
   else
      for(int e=0; e<imgCols_; e++)
         if(imgData_[e] != i2.imgData_[e])
            isEq = false;

   return isEq;
}

////////////////////////////////////////////////////////////////////////////////
// Most of CUDA stuff will be done in col-major format, but files store data
// in row-major format, which is why we switch the normal order of the loops
void cudaImageHost::readFile(string filename, int ncols, int nrows)
{
   resize(ncols, nrows);

   ifstream is(filename.c_str(), ios::in);
   for(int r=0; r<imgRows_; r++)
      for(int c=0; c<imgCols_; c++)
         is >> imgData_[c*imgRows_ + r];
   is.close();
}

////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::writeFile(string filename) const
{
   ofstream os(filename.c_str(), ios::out);
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
         os << imgData_[c*imgRows_+r] << " ";
      os << endl;
   }
   os.close();
}


////////////////////////////////////////////////////////////////////////////////
void cudaImageHost::printImage(void) const
{
   for(int r=0; r<imgRows_; r++)
   {
      for(int c=0; c<imgCols_; c++)
         cout << imgData_[c*imgRows_+r] << " ";
      cout << endl;
   }
}







