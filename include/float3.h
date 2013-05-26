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

__device__ float3 float3_negate ( float3 x )
{
  return make_float3 ( - x.x, - x.y, - x.z );
}

__device__ float3 float3_recip ( float3 x )
{
  return make_float3 ( 1.0 / x.x, 1.0 / x.y, 1.0 / x.z );
}

__device__ float float3_dot ( float3 a, float3 b )
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

#endif // CL_CUDA_FLOAT3_H_
