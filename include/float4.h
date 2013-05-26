/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_FLOAT4_H_
#define CL_CUDA_FLOAT4_H_

__device__ float4 float4_add ( float4 a, float4 b )
{
  return make_float4 ( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

__device__ float4 float4_sub ( float4 a, float4 b )
{
  return make_float4 ( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

__device__ float4 float4_scale ( float4 a, float k )
{
  return make_float4 ( a.x * k, a.y * k, a.z * k, a.w * k );
}

__device__ float4 float4_scale_flipped ( float k, float4 a )
{
  return float4_scale ( a, k );
}

__device__ float4 float4_scale_inverted ( float4 a, float k )
{
  return float4_scale ( a, 1.0 / k );
}

__device__ float4 float4_negate ( float4 x )
{
  return make_float4 ( - x.x, -x.y, -x.z, -x.w );
}

__device__ float4 float4_recip ( float4 x )
{
  return make_float4 ( 1.0 / x.x, 1.0 / x.y, 1.0 / x.z, 1.0 / x.w );
}

__device__ float float4_dot ( float4 a, float4 b )
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

#endif // CL_CUDA_FLOAT4_H_
