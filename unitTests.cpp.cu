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

#include "cudaConvUtilities.h.cu"
#include "cudaConvolution.h.cu"
#include "cudaMorphology.h.cu"
#include "ImageWorkbench.h.cu"

using namespace std;

unsigned int timer;

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

   cout << endl << "Executing GPU-accelerated convolution..." << endl;

   /////////////////////////////////////////////////////////////////////////////
   // Query the devices on the system and select the fastest
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
      cout << "No CUDA devices available.  Exiting." << endl;
      return -1;
	}

   // Fastest device automatically selected.  Can override below
   int fastestDeviceID = cutGetMaxGflopsDeviceId() ;
   //fastestDeviceID = 0;
   cudaSetDevice(fastestDeviceID);

   cudaDeviceProp gpuProp;
   cout << "CUDA-enabled devices on this system:  " << deviceCount <<  endl;
   for(int dev=0; dev<deviceCount; dev++)
   {
      cudaGetDeviceProperties(&gpuProp, dev); 
      char* devName = gpuProp.name;
      int mjr = gpuProp.major;
      int mnr = gpuProp.minor;
      int memMB = gpuProp.totalGlobalMem / (1024*1024);
      if( dev==fastestDeviceID )
         cout << "\t* ";
      else
         cout << "\t  ";

      printf("(%d) %20s (%d MB): \tCUDA Capability %d.%d \n", dev, devName, memMB, mjr, mnr);
   }

   /////////////////////////////////////////////////////////////////////////////
   //runMorphologyUnitTests();

   /////////////////////////////////////////////////////////////////////////////
   cudaThreadExit();

   //cutilExit(argc, argv);
}


void runConvolutionUnitTests(void)
{

}

////////////////////////////////////////////////////////////////////////////////
void runMorphologyUnitTests()
{
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

   // Also test a circle
   int  seCircD = 5; // D~Diameter
   int  seCircPixels = seCircD*seCircD;
   int  seCircBytes = seCircPixels*INT_SZ;
   int* seCirc = (int*)malloc(seCircBytes);
   int  seCircNZ = createBinaryCircle(seCirc, seCircD); // return #non-zero

   PrintArray(seCirc, 5, 5);

   int* devIn;
   int* devOut;
   int* devPsf;
   cudaMalloc((void**)&devIn,  imgBytes);
   cudaMalloc((void**)&devOut, imgBytes);
   cudaMalloc((void**)&devPsf, seCircBytes);
   cudaMemcpy(devIn,  imgIn,  imgBytes,    cudaMemcpyHostToDevice);
   cudaMemcpy(devPsf, seCirc, seCircBytes, cudaMemcpyHostToDevice);

   dim3 GRID(  32,  8, 1);
   dim3 BLOCK(  8, 32, 1);
   //Morph3x3_Dilate_Kernel<<<GRID,BLOCK>>>(devIn, devOut, imgW, imgH);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   cudaMemcpy(imgOut, devOut, imgBytes, cudaMemcpyDeviceToHost);
   
   WriteFile("ImageIn.txt", imgIn, imgW, imgH);
   WriteFile("ImageOut.txt", imgOut, imgW, imgH);



   free(imgIn);
   free(imgOut);
   //free(se17);
   //free(seRect);
   free(seCirc);
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
