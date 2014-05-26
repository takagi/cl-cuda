/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_DOUBLE4_H_
#define CL_CUDA_DOUBLE4_H_

__device__ double4 double4_add ( double4 a, double4 b )
{
  return make_double4 ( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

__device__ double4 double4_sub ( double4 a, double4 b )
{
  return make_double4 ( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

__device__ double4 double4_scale ( double4 a, double k )
{
  return make_double4 ( a.x * k, a.y * k, a.z * k, a.w * k );
}

__device__ double4 double4_scale_flipped ( double k, double4 a )
{
  return double4_scale ( a, k );
}

__device__ double4 double4_scale_inverted ( double4 a, double k )
{
  return double4_scale ( a, (double)1.0 / k );
}

__device__ double4 double4_negate ( double4 x )
{
  return make_double4 ( - x.x, -x.y, -x.z, -x.w );
}

__device__ double4 double4_recip ( double4 x )
{
  return make_double4 ( (double)1.0 / x.x, (double)1.0 / x.y, (double)1.0 / x.z, (double)1.0 / x.w );
}

__device__ double double4_dot ( double4 a, double4 b )
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

#endif // CL_CUDA_DOUBLE4_H_
