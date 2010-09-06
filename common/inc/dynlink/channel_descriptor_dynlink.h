/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#if !defined(__CHANNEL_DESCRIPTOR_H__)
#define __CHANNEL_DESCRIPTOR_H__

#if defined(__cplusplus)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "driver_types.h"
#include "cuda_runtime_api_dynlink.h"
#include "host_defines.h"
#include "vector_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template<class T> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
	return dyn::cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char>(void)
{
  int e = (int)sizeof(char) * 8;

#if __SIGNED_CHARS__
  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
#else
  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
#endif
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<signed char>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned char>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char1>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar1>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char2>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar2>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char4>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar4>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned short>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short1>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort1>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short2>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort2>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short4>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort4>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned int>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int1>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint1>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int2>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint2>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int4>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint4>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

#if !defined(__LP64__)

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned long>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long1>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong1>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long2>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong2>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long4>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong4>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

#endif /* !__LP64__ */

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float1>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float2>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float4>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

#endif /* __cplusplus */

#endif /* !__CHANNEL_DESCRIPTOR_H__ */
