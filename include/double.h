/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_DOUBLE_H_
#define CL_CUDA_DOUBLE_H_

__device__ double double_negate ( double x )
{
  return - x;
}

__device__  double double_recip ( double x )
{
  return (double)1.0 / x;
}

#endif // CL_CUDA_DOUBLE_H_
