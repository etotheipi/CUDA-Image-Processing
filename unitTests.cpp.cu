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
void runTimingTests(void);


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

   runCudaImageUnitTests();

   runMorphologyUnitTests();

   runWorkbenchUnitTests();

   runTimingTests();

   cudaThreadExit();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void runCudaImageUnitTests(void)
{
   cout << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Unit tests for CudaImage classes" << endl;

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
   cout << endl << "Executing morphology unit tests (no workbench)..." << endl;

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
   se17.printMask('.','0');
   cout << "Other tests using basic circular SE:" << endl;
   seCirc.printMask('.','0');

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


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void runWorkbenchUnitTests(void)
{
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Testing ImageWorkbench basic operations" << endl << endl;

   // Read the salt image from file
   cudaImageHost imgIn("salt256.txt", 256, 256);

   // Create a place to put the result
   cudaImageHost imgOut(64, 64);

   // A very unique SE for checking coordinate systems
   cudaImageHost se17("asymmPSF_17x17.txt", 17, 17);

   // Circular SE from utilities file
   int seCircD = 11;
   cudaImageHost seCirc(seCircD, seCircD);
   createBinaryCircle(seCirc.getDataPtr(), seCircD);

   // Check that rectangular SEs work, too
   int rectW = 9;
   int rectH = 5;
   cudaImageHost seRect(rectW, rectH);
   for(int r=0; r<rectH; r++)
      for(int c=0; c<rectW; c++)
         seRect(c, r) = 1;


   // The SEs are added to the static, master SE list in ImageWorkbench, and
   // are used by giving the index into that list (returned by addStructElt())
   cout << "Adding unique SE to list" << endl;
   se17.printMask();
   int seIdxUnique17 = ImageWorkbench::addStructElt(se17);

   cout << "Adding circular SE to list" << endl;
   seCirc.printMask();
   int seIdxCircle11 = ImageWorkbench::addStructElt(seCirc);

   cout << "Adding rectangular SE to list" << endl;
   seRect.printMask();
   int seIdxRect9x5  = ImageWorkbench::addStructElt(seRect);
   
   cudaImageDevice::calculateDeviceMemoryUsage(true);  // printToStdOut==true


   /////////////////////////////////////////////////////////////////////////////
   // Let's start testing ImageWorkbench
   /////////////////////////////////////////////////////////////////////////////
   // Create the workbench, which copies the image into device memory
   ImageWorkbench theIwb(imgIn);

   // Start by simply fetching the unmodified image (sanity check)
   cout << "Copying unaltered image back to host for verification" << endl;
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench1_In.txt");
   
   // Dilate by the circle
   cout << "Dilating with 11x11 circle" << endl;
   theIwb.Dilate(seIdxCircle11);
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench2_DilateCirc.txt");

   // We Erode the image now, but with the basic 3x3
   cout << "Performing simple 3x3 erode" << endl;
   theIwb.Erode();
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench3_Erode3.txt");

   // We Erode the image now, but with the basic 3x3
   cout << "Try a closing operation" << endl;
   theIwb.Close(seIdxCircle11);
   theIwb.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench4_Close.txt");

   // We now test subtract by eroding an image w/ 3x3 and subtracting from original
   // Anytime we manually select src/dst for image operations, make sure we end up
   // with the final result in buffer A, or in buffer B with a a call to flipBuffers()
   // to make sure that our input/output locations are consistent
   ImageWorkbench iwb2(imgIn);
   cout << "Testing subtract kernel" << endl;
   
   iwb2.Dilate();
   iwb2.Dilate();
   iwb2.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench5a_dilated.txt");

   iwb2.Erode(A, 1);  // put result in buffer 1, don't flip
   iwb2.copyBufferToHost(1, imgOut);
   imgOut.writeFile("Workbench5b_erode.txt");
   
   iwb2.Subtract(1, A, A);
   iwb2.copyBufferToHost(A, imgOut);
   imgOut.writeFile("Workbench5c_subtract.txt");

   cudaImageHost cornerDetect(3,3);
   cornerDetect(0,0) = -1;  cornerDetect(1,0) = -1;  cornerDetect(2,0) = 0;
   cornerDetect(0,1) = -1;  cornerDetect(1,1) =  1;  cornerDetect(2,1) = 1;
   cornerDetect(0,2) =  0;  cornerDetect(1,2) =  1;  cornerDetect(2,2) = 0;
   int seIdxCD = ImageWorkbench::addStructElt(cornerDetect);
   iwb2.FindAndRemove(seIdxCD);
   iwb2.copyBufferToHost(imgOut);
   imgOut.writeFile("Workbench5d_findandrmv.txt");

   cout << endl << "Checking device memory usage so far: " << endl;
   cudaImageDevice::calculateDeviceMemoryUsage(true);  // printToStdOut==true

   /////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////
   // With a working workbench, we can finally SOLVE A MAZE !!
   cout << endl << "Time to solve a maze! " << endl << endl;
   cudaImageHost mazeImg("elephantmaze.txt", 512, 512);
   ImageWorkbench iwbMaze(mazeImg);

   // Morph-close the image [for fun, not necessary], write it to file for ref
   iwbMaze.Close();  
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("Maze1_In.txt");

   // Start thinning
   cout << "\tThinning sweep 2x" << endl;
   iwbMaze.ThinningSweep();
   iwbMaze.ThinningSweep();
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("Maze2_Thin2x.txt");


   // Finish thinning by checking when the image is no longer changing
   cout << "\tThinning sweep til complete" << endl;
   int thinOps = 2;
   int diff=-1;
   while(diff != 0)
   {
      iwbMaze.ThinningSweep();
      diff = iwbMaze.CountChanged();
      thinOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("Maze3_ThinComplete.txt");

   cout << "\tPruning sweep 1-5" << endl;
   int pruneOps = 0;
   for(int i=0; i<5; i++)
   {
      iwbMaze.PruningSweep();
      pruneOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("Maze4_Prune5x.txt");

   cout << "\tPruning sweep 6-20" << endl;
   for(int i=0; i<15; i++)
   {
      iwbMaze.PruningSweep();
      pruneOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("Maze5_Prune20x.txt");

   diff=-1;
   cout << "\tPruning sweep until complete" << endl;
   while(diff != 0)
   {
      iwbMaze.PruningSweep();
      diff = iwbMaze.CountChanged();
      pruneOps++;
   }
   iwbMaze.copyBufferToHost(imgOut);
   imgOut.writeFile("Maze6_PruneComplete.txt");

   int totalHomOps = 8*(thinOps + pruneOps);
   cout << "Finished the maze!  Total operations: " << endl
        << "\t" << thinOps  << " thinning sweeps and " << endl
        << "\t" << pruneOps << " pruning sweeps" << endl
        << "\tTotal of " << totalHomOps << " HitOrMiss operations and the same "
        << "number of subtract operations" << endl << endl;


   // Check to see how much device memory we're using right now
   cudaImageDevice::calculateDeviceMemoryUsage(true);  // printToStdOut==true

   cout << "Finished IWB testing!" << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;

}


#define TIME(a) \
do { \
         \
} while(0);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void runTimingTests(void)
{
   cout << endl << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Executing timing tests..." << endl;

   // Make sure all previous GPU ops are done
   cudaThreadSynchronize();

   // First measure simple allocations and copying of host/device memory

   float gputime, cputime;

   cout << "\tNow allocate a 4096x4096 images and move them around:" << endl;
   gpuStartTimer();
   cudaImageDevice deviceBigImg(4096,4096);
   gputime = gpuStopTimer();
   printf("\t\tAllocating 64MB in device memory took %0.2f ms (%.0f MB/s)\n", gputime, 64000.0f/gputime);

   cpuStartTimer();
   cudaImageHost hostBigImg(4096,4096);
   cputime = cpuStopTimer();

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

   
   // We previously timed the host allocation but didn't report it
   printf("\t\tAllocating 64MB in HOST memory took %0.2f ms (%.0f MB/s)\n", cputime, 64000.0f/cputime);

   cudaImageHost moreHostData(4096,4096);
   cpuStartTimer();
   moreHostData = hostBigImg;
   cputime = cpuStopTimer();
   printf("\t\tCopying 64MB within HOST took %0.2f ms (%.0f MB/s)\n", cputime, 64000.0f/cputime);


   // First we do elaborate timings on raw kernel functions using direct memory
   // locations.  Then we will do the same thing with the workbench and see how 
   // much overhead there is.  I expect there will be virtually no overhead, but
   // I won't know til I test it.

   cout << endl << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Timing a variety of morphological median calculations..." << endl;
   int NITER=10;

   vector<cudaImageDevice> circ(8);
   circ[0].copyFromHost(createBinaryCircle(3));
   circ[1].copyFromHost(createBinaryCircle(5));
   circ[2].copyFromHost(createBinaryCircle(7));
   circ[3].copyFromHost(createBinaryCircle(9));
   circ[4].copyFromHost(createBinaryCircle(11));
   circ[5].copyFromHost(createBinaryCircle(13));
   circ[6].copyFromHost(createBinaryCircle(15));
   circ[7].copyFromHost(createBinaryCircle(17));

   int testSizes[5] = {256, 512, 1024, 2048};
   for(int test=0; test<4; test++)
   {
      int size = testSizes[test];
      int sizesq = size*size;

      dim3 BLOCK(8, 32, 1);
      dim3 GRID(size/BLOCK.x, size/BLOCK.y, 1);

      cudaImageHost   imgHost(size,size);
      cudaImageDevice imgDeviceIn(size,size);
      cudaImageDevice imgDeviceOut(size,size);
      cudaImageDevice imgTemp1(size,size);
      cudaImageDevice imgTemp2(size,size);

      // Put some data in the host image for fun
      for(int i=0; i<size*size; i++)
         imgHost[i] = i%2;

      imgDeviceIn.copyFromHost(imgHost);

      // Remember, for this test, only pointers
      int* devIn  = imgDeviceIn.getDataPtr();
      int* devOut = imgDeviceOut.getDataPtr();
      int* devTemp1 = imgTemp1.getDataPtr();
      int* devTemp2 = imgTemp2.getDataPtr();

      // First test optimized 3x3 kernels
      gpuStartTimer();
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         Morph3x3_Dilate_Kernel<<<GRID,BLOCK>>>(devIn, devOut, size, size);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tMorph %4dx%4d image with optimized 3x3:     (real ms, gpu ms, FPS) = (%.2f, %.2f, %.1f FPS)\n",
            size, size, cputime, gputime, 1000.0/cputime);

      for(int seIdx=0; seIdx<8; seIdx++)
      {
         int* se     = circ[seIdx].getDataPtr();
         int  seDiam = circ[seIdx].numCols();
         int  seRad  = seDiam / 2;

         gpuStartTimer();
         cpuStartTimer();
         for(int i=0; i<NITER; i++)
            Morph_Generic_Kernel<<<GRID,BLOCK>>>(devIn, devOut, size, size, se, seRad, seRad, 0);
         cudaThreadSynchronize();
         cputime = cpuStopTimer()/NITER;
         gputime = gpuStopTimer()/NITER;
      
         printf("\tMorph %4dx%4d image with %2dx%2d struct elt:  (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n",
               size, size, seDiam, seDiam, cputime, gputime, 1000.0/cputime);
      }

      // Now test the linear-array kernels
      dim3 BLOCK1D(256, 1, 1);
      dim3 GRID1D(sizesq/256, 1, 1);
      dim3 GRID1Dhalf(sizesq/512, 1, 1);
      gpuStartTimer();
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         Mask_Union_Kernel<<<GRID1D,BLOCK1D>>>(devIn, devOut, devOut);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tMask  %-16s two %4dx%4d images:  (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n", 
                     "UNION", size, size, cputime, gputime, 1000/cputime);

      gpuStartTimer();
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         Mask_Intersect_Kernel<<<GRID1D,BLOCK1D>>>(devIn, devOut, devOut);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tMask  %-16s two %4dx%4d images:  (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n", 
                     "INTERSECT", size, size, cputime, gputime, 1000/cputime);

      gpuStartTimer();
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         Mask_Subtract_Kernel<<<GRID1D,BLOCK1D>>>(devIn, devOut, devOut);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tMask  %-16s two %4dx%4d images:  (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n", 
                     "SUBTRACT", size, size, cputime, gputime, 1000/cputime);

      gpuStartTimer();
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         Mask_Invert_Kernel<<<GRID1D,BLOCK1D>>>(devIn, devOut);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tMask  %-16s one %4dx%4d image:   (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n", 
                     "INVERT", size, size, cputime, gputime, 1000/cputime);

      gpuStartTimer();
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         Mask_Difference_Kernel<<<GRID1D,BLOCK1D>>>(devIn, devOut, devOut);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tMask  %-16s two %4dx%4d images:  (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n", 
                     "DIFFERENCE", size, size, cputime, gputime, 1000/cputime);

      gpuStartTimer();
      cpuStartTimer();
      int k;
      for(int i=0; i<NITER; i++)
         k = Image_Sum(devIn, devTemp1, devTemp2, sizesq);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tImage %-16s one %4dx%4d image:   (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n", 
                     "REDUCTION SUM", size, size, cputime, gputime, 1000/cputime);
      cout << "\t\tsum=" << k <<  " (should be " << size*size/2 << ")" << endl;

      cout << endl;
      cout << endl;

   }
   cout << endl << endl;

   /////////////////////////////////////////////////////////////////////////////
   cout << endl << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << "***Timing tests on different block sizes, 1024x1024 image with 7x7 SE" << endl << endl;

   int size = 1024;
   int seDiam = 7;
   dim3 BLOCK;
   dim3 GRID;
   cudaImageHost   imgHost(size,size);
   cudaImageDevice in1024(size,size);
   cudaImageDevice out1024(size,size);
   cudaImageDevice se7(createBinaryCircle(seDiam));


   vector<dim3> BLOCKvect(0);
   BLOCKvect.push_back( dim3( 16, 16, 1) );
   BLOCKvect.push_back( dim3( 32,  8, 1) );
   BLOCKvect.push_back( dim3(  8, 32, 1) );
   BLOCKvect.push_back( dim3( 16, 32, 1) );
   BLOCKvect.push_back( dim3( 32, 16, 1) );
   BLOCKvect.push_back( dim3( 16,  8, 1) );
   BLOCKvect.push_back( dim3(  8, 16, 1) );
   BLOCKvect.push_back( dim3(  8,  8, 1) );
   
   for(int bs=0; bs<(int)BLOCKvect.size(); bs++)
   {
      dim3 BLOCK = BLOCKvect[bs];
      dim3 GRID(size/BLOCK.x, size/BLOCK.y, 1);
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         Morph_Generic_Kernel<<<GRID,BLOCK>>>(in1024, out1024, size, size, se7, seDiam/2, seDiam/2, 0);
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      printf("\tBlock size = (%2d,%2d), Grid size = (%3d,%3d), CPU Timing:  %2.2f ms -- %3.1f FPS\n",
            BLOCK.x, BLOCK.y, GRID.x, GRID.y, cputime, 1000/cputime);
   }




   // Now test ImageWorkbench, but we don't need as many tests
   // We just want to confirm that there isn't any crazy overhead using IWB
   // There shouldn't be, but we won't know til we test it
   cout << endl << endl;
   cout << "****************************************";
   cout << "***************************************" << endl;
   cout << endl << "***Test a few of the same operations with the workbench" << endl;
   int seIdx[2];
   seIdx[0] = ImageWorkbench::addStructElt(createBinaryCircle(7));
   seIdx[1] = ImageWorkbench::addStructElt(createBinaryCircle(15));

   for(int test=1; test<5; test+=2)
   {
      int size = testSizes[test];
      cudaImageHost   imgHost(size,size);
      for(int i=0; i<size*size; i++)
         imgHost[i] = i%2;

      ImageWorkbench testIwb(imgHost);

      gpuStartTimer();
      cpuStartTimer();
      for(int i=0; i<NITER; i++)
         testIwb.Median();
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tWorkbench %4dx%4d with optimized 3x3:     (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n",
            size, size, cputime, gputime, 1000.0/cputime);

      for(int s=0; s<2; s++)
      {
         gpuStartTimer();
         cpuStartTimer();
         for(int i=0; i<NITER; i++)
            testIwb.Median(seIdx[s]);
         cudaThreadSynchronize();
         cputime = cpuStopTimer()/NITER;
         gputime = gpuStopTimer()/NITER;

         int seSize = ImageWorkbench::getStructEltPtr(seIdx[s])->numCols();
         printf("\tWorkbench %4dx%4d with %2dx%2d struct elt:  (real ms, gpu ms, FPS) = (%0.2f, %0.2f; %.1f FPS)\n",
               size, size, seSize, seSize, cputime, gputime, 1000.0/cputime);
      }

      ImageWorkbench testIwb2(imgHost);

      gpuStartTimer();
      cpuStartTimer();
      int k;
      for(int i=0; i<NITER; i++)
         k = testIwb2.SumImage();
      cudaThreadSynchronize();
      cputime = cpuStopTimer()/NITER;
      gputime = gpuStopTimer()/NITER;
      printf("\tReduction SUM on a %4dx%4d image:  (real ms, gpu ms, FPS) = (%.2f, %.2f; %.1f FPS)\n", 
                     size, size, cputime, gputime, 1000/cputime);
      cout << "\tsum=" << k << "   (should be " << size*size/2 << ")" << endl;
      cout << endl;    
   }
   cout << endl << endl;
   cudaImageDevice::calculateDeviceMemoryUsage(true);


}



