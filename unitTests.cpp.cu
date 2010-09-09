/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <stopwatch.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <stopwatch.h>
#include <cmath>

#include "cudaImageHost.h"
#include "cudaImageDevice.h.cu"
#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"
#include "cudaMorphology.h.cu"
#include "ImageWorkbench.h.cu"

using namespace std;

unsigned int timer;

int  runDevicePropertiesQuery(void);
void runCudaImageUnitTests(void);
void runConvolutionUnitTests(void);
void runMorphologyUnitTests(void);
void runWorkbenchUnitTests(void);


////////////////////////////////////////////////////////////////////////////////
//
// Program main
//
// TODO:  Remove the CUTIL calls so libcutil is not required to compile/run
//
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
   runDevicePropertiesQuery();

   // This spits out a lot of data, but it is informative so I prefer to keep
   // it here.  Comment it out if desired.
   runCudaImageUnitTests();

   runMorphologyUnitTests();

   cudaThreadExit();
}

/////////////////////////////////////////////////////////////////////////////
// Query the devices on the system and select the fastest (or override the
// selectedDevice variable to choose your own
int runDevicePropertiesQuery(void)
{
   cout << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Device query and selection:" << endl;
   int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
   {
		cout << "cudaGetDeviceCount() FAILED." << endl;
      cout << "CUDA Driver and Runtime version may be mismatched.\n";
      return -1;
	}

   // Check to make sure we have at least on CUDA-capable device
   if( deviceCount == 0)
   {
      cout << "No CUDA devices available." << endl;
      return -1;
	}

   // Fastest device automatically selected.  Can override below
   int selectedDevice = cutGetMaxGflopsDeviceId() ;
   //selectedDevice = 0;
   cudaSetDevice(selectedDevice);

   cudaDeviceProp gpuProp;
   cout << "CUDA-enabled devices on this system:  " << deviceCount <<  endl;
   for(int dev=0; dev<deviceCount; dev++)
   {
      cudaGetDeviceProperties(&gpuProp, dev); 
      char* devName = gpuProp.name;
      int mjr = gpuProp.major;
      int mnr = gpuProp.minor;
      int memMB = gpuProp.totalGlobalMem / (1024*1024);
      if( dev==selectedDevice )
         cout << "\t* ";
      else
         cout << "\t  ";

      printf("(%d) %20s (%d MB): \tCUDA Capability %d.%d \n", dev, devName, memMB, mjr, mnr);
   }

   cout << "****************************************";
   cout << "***************************************" << endl;
   return selectedDevice;
}

void runCudaImageUnitTests(void)
{
   cout << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Test CudaImage classes and GPU allocation/copy speeds" << endl;

   // Allocate some memory the "old" way
   int  d = 5; // D~Diameter
   int  testPixels = d*d;
   int  testBytes = testPixels*INT_SZ;
   int* test = (int*)malloc(testBytes);
   for(int i=0; i<testPixels; i++)
      test[i] = i;

   cudaImageHost h_img(test, d, d);
   cudaImageHost h_img1(test, d, d);
   free(test);


   printf("\t%-50s", "Testing ImageHost basic constructor");
   printf("Passed?  %d \n", (int)(h_img1==h_img));

   printf("\t%-50s", "Testing ImageHost file I/O");
   h_img1.writeFile("test5x5.txt");
   h_img1.readFile("test5x5.txt", 5, 5);
   printf("Passed?  %d \n", (int)(h_img1==h_img));

   printf("\t%-50s", "Testing ImageHost copy constructor");
   cudaImageHost h_img2(h_img1);
   printf("Passed?  %d \n", (int)(h_img2==h_img));

   printf("\t%-50s", "Testing ImageHost operator=()");
   cudaImageHost h_img3 = h_img2;
   printf("Passed?  %d \n", (int)(h_img3==h_img));

   printf("\t%-50s", "Testing ImageHost op= with diff img sizes");
   h_img3 = cudaImageHost(6,6);
   h_img3 = h_img2;
   printf("Passed?  %d \n", (int)(h_img3==h_img));

   printf("\t%-50s","Testing ImageDevice constructor and copyToHost");
   cudaImageDevice d_img1(h_img3);
   cudaImageHost h_img4(d, d);
   d_img1.copyToHost(h_img4);
   printf("Passed?  %d \n", (int)(h_img4==h_img));
   
   printf("\t%-50s","Testing ImageDevice copyFromHost and copyToHost");
   cudaImageDevice d_img2;
   d_img2.copyFromHost(h_img4);
   cudaImageHost h_img5(d,d);
   d_img2.copyToHost(h_img5);
   printf("Passed?  %d \n", (int)(h_img5==h_img));

   printf("\t%-50s","Testing ImageDevice another constructor");
   cudaImageDevice d_img3(d, d);
   d_img3.copyFromHost(h_img3);
   d_img3.copyToHost(h_img5);
   printf("Passed?  %d \n", (int)(h_img5==h_img));

   printf("\t%-50s","Testing ImageDevice one more constructor");
   cudaImageDevice d_img4(d+1, d+1);
   d_img4.copyFromHost(h_img3);
   d_img4.copyToHost(h_img5);
   printf("Passed?  %d \n", (int)(h_img5==h_img));

   printf("\t%-50s","Testing ImageDevice Device2Device");
   cudaImageDevice d_img5(d+1, d+1);
   d_img5.copyFromDevice(h_img4);
   d_img5.copyToHost(h_img5);
   printf("Passed?  %d \n", (int)(h_img5==h_img));

   cout << endl << endl;

   float gputime;

   cout << "\tNow allocate a 4096x4096 image and move it around:" << endl;
   gpuStartTimer();
   cudaImageDevice deviceBigImg(4096,4096);
   gputime = gpuStopTimer();
   printf("\t\tAllocating 64MB in device memory took %0.2f ms (%.0f MB/s)\n", gputime, 64000.0f/gputime);

   cudaImageHost hostBigImg(4096,4096);
   gpuStartTimer();
   deviceBigImg.copyFromHost(hostBigImg);
   gputime = gpuStopTimer();
   printf("\t\tCopying 64MB from HOST to DEVICE took %0.2f ms (%.0f MB/s)\n", gputime, 64000.0f/gputime);

   gpuStartTimer();
   deviceBigImg.copyToHost(hostBigImg);
   gputime = gpuStopTimer();
   printf("\t\tCopying 64MB from DEVICE to HOST took %0.2f ms (%.0f MB/s)\n", gputime, 64000.0f/gputime);

   cudaImageDevice copyOfBigImg(4096, 4096);
   gpuStartTimer();
   deviceBigImg.copyToDevice(copyOfBigImg);
   gputime = gpuStopTimer();
   printf("\t\tCopying 64MB within DEVICE took %0.2f ms (%.0f MB/s)\n\n\n", gputime, 64000.0f/gputime);

   cout << "\tCheck current device memory usage:" << endl;
   cudaImageDevice::calculateDeviceMemoryUsage(true);

   cout << "****************************************";
   cout << "***************************************" << endl;
}

void runConvolutionUnitTests(void)
{

}

////////////////////////////////////////////////////////////////////////////////
void runMorphologyUnitTests()
{

   cout << endl << "Executing morphology unit tests..." << endl;

   /////////////////////////////////////////////////////////////////////////////
   // Allocate host memory and read in the test image from file
   /////////////////////////////////////////////////////////////////////////////
   unsigned int imgW  = 256;
   unsigned int imgH  = 256;
   unsigned int nPix  = imgW*imgH;
   string fn("salt256.txt");

   printf("\nTesting morphology operations on %dx%d mask.\n", imgW,imgH);
   cout << "Reading mask from " << fn.c_str() << endl << endl;

   cudaImageHost imgIn(fn, imgW, imgH);
   cudaImageHost imgOut(imgW, imgH);

   imgIn.writeFile("ImageIn.txt");


   // A very unique SE for checking coordinate systems
   int se17H = 17;
   int se17W = 17;
   cudaImageHost se17("asymmPSF_17x17.txt", se17W, se17H);

   // Circular SE from utilities file
   int seCircD = 5;
   cudaImageHost seCirc(seCircD, seCircD);
   int seCircNZ = createBinaryCircle(seCirc.getDataPtr(), seCircD); // return #non-zero

   // Display the two SEs
   cout << "Using the unique, 17x17 structuring element:" << endl;
   se17.printImage();
   cout << "Other tests using basic circular SE:" << endl;
   seCirc.printImage();

   // Allocate Device Memory
   cudaImageDevice devIn(imgIn);
   cudaImageDevice devPsf(se17);
   cudaImageDevice devOut(imgW, imgH);

   cudaImageDevice::calculateDeviceMemoryUsage(true);


   int bx = 8;
   int by = 32;
   int gx = imgW/bx;
   int gy = imgH/by;
   dim3 BLOCK1D( bx*by, 1, 1);
   dim3 GRID1D(  nPix/(bx*by), 1, 1);
   dim3 BLOCK2D( bx, by, 1);
   dim3 GRID2D(  gx, gy, 1);

   /////////////////////////////////////////////////////////////////////////////
   // TEST THE GENERIC/UNIVERSAL MORPHOLOGY OPS
   // Non-zero elts = 134, so use -133 for dilate
   Morph_Generic_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH, 
                                                devPsf, se17H/2, se17W/2, -133);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("ImageDilate.txt");

   // Non-zero elts = 134, so use 134 for erod
   Morph_Generic_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH, 
                                                devPsf, se17H/2, se17W/2, 134);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("ImageErode.txt");

   /////////////////////////////////////////////////////////////////////////////
   // We also need to verify that the 3x3 optimized functions work
   Morph3x3_Dilate_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("Image3x3_dilate.txt");

   Morph3x3_Erode4connect_Kernel<<<GRID2D,BLOCK2D>>>(devIn, devOut, imgW, imgH);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devOut.copyToHost(imgOut);
   imgOut.writeFile("Image3x3_erode.txt");

   Morph3x3_Thin8_Kernel<<<GRID2D,BLOCK2D>>>(devOut, devIn, imgW, imgH);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   devIn.copyToHost(imgIn);
   imgIn.writeFile("Image3x3_erode_thin.txt");
   /////////////////////////////////////////////////////////////////////////////
}


void runWorkbenchUnitTests(void)
{
   /*
   /////////////////////////////////////////////////////////////////////////////
   // Allocate host memory and read in the test image from file
   /////////////////////////////////////////////////////////////////////////////
   unsigned int imgW  = 256;
   unsigned int imgH  = 256;
   unsigned int nPix  = imgH*imgW;
   unsigned int imgBytes = nPix*INT_SZ;
   int* imgIn  = (int*)malloc(imgBytes);
   int* imgOut = (int*)malloc(imgBytes);
   string fn("salt256.txt");

   cout << endl;
   printf("\nTesting morphology operations on %dx%d mask.\n", imgW,imgH);
   cout << "Reading mask from " << fn.c_str() << endl;
   ReadFile(fn, imgIn, imgW, imgH);



   /////////////////////////////////////////////////////////////////////////////
   // Create a bunch of structuring elements to test
   //
   // All the important 3x3 SEs are "hardcoded" into dedicated functions 
   // See all the CREATE_3X3_MORPH_KERNEL/CREATE_IWB_3X3_FUNCTION calls in .cu
   /////////////////////////////////////////////////////////////////////////////
   // A very unique 17x17 to test coord systems
   int  se17W = 17;
   int  se17H = 17;
   int  se17Pixels = se17W*se17H;
   int  se17Bytes = se17Pixels * INT_SZ;
   int* se17 = (int*)malloc(se17Bytes);
   ReadFile("asymmPSF_17x17.txt",   se17,  se17W,  se17H);

   // Test a rectangular SE
   int  seRectW = 9;
   int  seRectH = 5;
   int  seRectPixels = seRectW*seRectH;
   int  seRectBytes  = seRectPixels * INT_SZ;
   int* seRect = (int*)malloc(seRectBytes);
   for(int i=0; i<seRectPixels; i++)
      seRect[i] = 1;

   // Also test a circle
   int  seCircD = 11; // D~Diameter
   int  seCircPixels = seCircD*seCircD;
   int  seCircBytes = seCircPixels*INT_SZ;
   int* seCirc = (int*)malloc(seCircBytes);
   int  seCircNZ = createBinaryCircle(seCirc, seCircD); // return #non-zero

   // Add the structuring elements to the master SE list, which copies 
   // them into device memory.  Note that you need separate SEs for
   // erosion and dilation, even if they are the same img-data (target
   // sum is different)
   //int seIdxUnique17x17  = ImageWorkbench::addStructElt(se17,   se17W,   se17H  );
   //int seIdxRect9x5      = ImageWorkbench::addStructElt(seRect, seRectW, seRectH);
   //int seIdxCircle11x11  = ImageWorkbench::addStructElt(seCirc, seCircD, seCircD);
   int seIdxUnique17x17 = 0;
   int seIdxRect9x5     = 0;
   int seIdxCircle11x11 = 0;
   

   /////////////////////////////////////////////////////////////////////////////
   // Let's start testing ImageWorkbench
   /////////////////////////////////////////////////////////////////////////////

   // Create the workbench, which copies the image into device memory
   ImageWorkbench theIwb(imgIn, imgW, imgH);

   dim3 bsize = theIwb.getBlockSize();
   dim3 gsize = theIwb.getGridSize();
   printf("Using the following kernel geometry for morphology operations:\n");
   printf("\tBlock Size = (%d, %d, %d) threads\n", bsize.x, bsize.y, bsize.z);
   printf("\tGrid Size  = (%d, %d, %d) blocks\n ", gsize.x, gsize.y, gsize.z);

   // Start by simply fetching the unmodified image (sanity check)
   theIwb.fetchResult(imgOut);
   WriteFile("Image1_Orig.txt", imgOut, imgW, imgH);
   
   // Dilate by the 17x17
   theIwb.Dilate(seIdxUnique17x17);
   theIwb.fetchResult(imgOut);
   WriteFile("Image2_Dilate17.txt", imgOut, imgW, imgH);

   // We Erode the image now, which means it's actually been "closed"
   theIwb.Erode(seIdxUnique17x17);
   theIwb.fetchResult(imgOut);
   WriteFile("Image3_Close17.txt", imgOut, imgW, imgH);

   // Dilate with rectangle
   theIwb.Dilate(seIdxRect9x5);
   theIwb.fetchResult(imgOut);
   WriteFile("Image4_DilateRect.txt", imgOut, imgW, imgH);

   // Try a thinning sweep on the dilated image (8 findandremove ops) 
   theIwb.ThinningSweep();
   theIwb.fetchResult(imgOut);
   WriteFile("Image6_ThinSw1.txt", imgOut, imgW, imgH);

   // Again...
   theIwb.ThinningSweep();
   theIwb.fetchResult(imgOut);
   WriteFile("Image7_ThinSw2.txt", imgOut, imgW, imgH);

   // And again...
   theIwb.ThinningSweep();
   theIwb.fetchResult(imgOut);
   WriteFile("Image8_ThinSw3.txt", imgOut, imgW, imgH);

   // Check to see how much device memory we're using right now
   ImageWorkbench::calculateDeviceMemUsage(true);  // printToStdOut==true

   free(imgIn);
   free(imgOut);
   free(se17);
   free(seRect);
   free(seCirc);
   */
}
