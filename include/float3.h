/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_FLOAT3_H_
#define CL_CUDA_FLOAT3_H_

__device__ float3 float3_add ( float3 a, float3 b )
{
  return make_float3 ( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ float3 float3_sub ( float3 a, float3 b )
{
  return make_float3 ( a.x - b.x, a.y - b.y, a.z - b.z );
}

__device__ float3 float3_scale ( float3 a, float k )
{
  return make_float3 ( a.x * k, a.y * k, a.z * k );
}

__device__ float3 float3_scale_flipped ( float k, float3 a )
{
  return float3_scale ( a, k );
}

__device__ float3 float3_scale_inverted ( float3 a, float k )
{
  return float3_scale ( a, 1.0 / k );
}

#endif
