/**
   This file is a part of cl-cuda project.
   Copyright (c) 2019 Juan M. Bello-Rivas (jbellorivas@rigetti.com)
 */

#ifndef CL_CUDA_FLOAT2_H_
#define CL_CUDA_FLOAT2_H_

__device__ float2 float2_add ( float2 a, float2 b )
{
  return make_float2 ( a.x + b.x, a.y + b.y );
}

__device__ float2 float2_sub ( float2 a, float2 b )
{
  return make_float2 ( a.x - b.x, a.y - b.y );
}

__device__ float2 float2_scale ( float2 a, float k )
{
  return make_float2 ( a.x * k, a.y * k );
}

__device__ float2 float2_scale_flipped ( float k, float2 a )
{
  return float2_scale ( a, k );
}

__device__ float2 float2_scale_inverted ( float2 a, float k )
{
  return float2_scale ( a, 1.0 / k );
}

__device__ float2 float2_negate ( float2 x )
{
  return make_float2 ( - x.x, - x.y );
}

__device__ float2 float2_recip ( float2 x )
{
  return make_float2 ( 1.0 / x.x, 1.0 / x.y );
}

__device__ float float2_dot ( float2 a, float2 b )
{
  return a.x * b.x + a.y * b.y;
}

#endif // CL_CUDA_FLOAT2_H_
