/**
   This file is a part of cl-cuda project.
   Copyright (c) 2019 Juan M. Bello-Rivas (jbellorivas@rigetti.com)
 */

#ifndef CL_CUDA_DOUBLE2_H_
#define CL_CUDA_DOUBLE2_H_

__device__ double2 double2_add ( double2 a, double2 b )
{
  return make_double2 ( a.x + b.x, a.y + b.y );
}

__device__ double2 double2_sub ( double2 a, double2 b )
{
  return make_double2 ( a.x - b.x, a.y - b.y );
}

__device__ double2 double2_scale ( double2 a, float k )
{
  return make_double2 ( a.x * k, a.y * k );
}

__device__ double2 double2_scale_flipped ( float k, double2 a )
{
  return double2_scale ( a, k );
}

__device__ double2 double2_scale_inverted ( double2 a, float k )
{
  return double2_scale ( a, 1.0 / k );
}

__device__ double2 double2_negate ( double2 x )
{
  return make_double2 ( - x.x, - x.y );
}

__device__ double2 double2_recip ( double2 x )
{
  return make_double2 ( 1.0 / x.x, 1.0 / x.y );
}

__device__ float double2_dot ( double2 a, double2 b )
{
  return a.x * b.x + a.y * b.y;
}

#endif // CL_CUDA_DOUBLE2_H_
