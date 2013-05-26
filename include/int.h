/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_INT_H_
#define CL_CUDA_INT_H_

__device__ int int_negate ( int x )
{
  return -x;
}

__device__ int int_recip ( int x )
{
  return 1 / x;
}

#endif // CL_CUDA_INT_H_
