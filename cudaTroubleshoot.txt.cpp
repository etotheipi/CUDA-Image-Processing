This file will be notes about various issues I've encountered while trying to 
compile or run CUDA code.  It's a complete pain in the ass when you get runtime
errors, because it doesn't tell you where.  Frequently, it doesn't tell you 
anything useful.  After having run into a few of these multiple times, and 
never knowing for sure what the problem was or how I fixed it, I figured that
keeping a journal of them like this would help.

-----
Executed fine, but results are empty

*** Check your block/grid size

*** Make sure you pass DEVICE MEMORY pointers to the kernel functions.  Numeric
    values get passed through the kernel invocation, but the device cannot read
    host memory.  You knew this already, but it's easy to use the wrong pointer
    when calling the kernel functions

-----
"Kernel execution failed : invalid argument."


*** Block/grid size could be bad.  For compute capability 2.0+, I like to keep
    the block size at 256 (8x32 instead of 16x16 helps reduce bank conflicts). 
    I think you can't go above 512, though. 

*** Seems that if you don't allocate the device memory properly, you get
    this error.  For instance, I cudaMalloc'd for 5x5, then cudaMemcpy'd
    for a 17x17, and I didn't get an error until I tried to execute the
    kernel



