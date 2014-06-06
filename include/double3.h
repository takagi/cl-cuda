/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_DOUBLE3_H_
#define CL_CUDA_DOUBLE3_H_

__device__ double3 double3_add ( double3 a, double3 b )
{
  return make_double3 ( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ double3 double3_sub ( double3 a, double3 b )
{
  return make_double3 ( a.x - b.x, a.y - b.y, a.z - b.z );
}

__device__ double3 double3_scale ( double3 a, double k )
{
  return make_double3 ( a.x * k, a.y * k, a.z * k );
}

__device__ double3 double3_scale_flipped ( double k, double3 a )
{
  return double3_scale ( a, k );
}

__device__ double3 double3_scale_inverted ( double3 a, double k )
{
  return double3_scale ( a, (double)1.0 / k );
}

__device__ double3 double3_negate ( double3 x )
{
  return make_double3 ( - x.x, - x.y, - x.z );
}

__device__ double3 double3_recip ( double3 x )
{
  return make_double3 ( (double)1.0 / x.x, (double)1.0 / x.y, (double)1.0 / x.z );
}

__device__ double double3_dot ( double3 a, double3 b )
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

#endif // CL_CUDA_DOUBLE3_H_
