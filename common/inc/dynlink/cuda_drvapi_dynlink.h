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
 
 
#ifndef __cuda_drvapi_dynlink_h__
#define __cuda_drvapi_dynlink_h__

#include "cuda_drvapi_dynlink_cuda.h"

#if defined(CUDA_INIT_D3D9)||defined(CUDA_INIT_D3D10)||defined(CUDA_INIT_D3D11)
#include "cuda_drvapi_dynlink_d3d.h"
#endif

#ifdef CUDA_INIT_OPENGL
#include "cuda_drvapi_dynlink_gl.h"
#endif

#endif //__cuda_drvapi_dynlink_h__
