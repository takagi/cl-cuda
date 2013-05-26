/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_FLOAT_H_
#define CL_CUDA_FLOAT_H_

__device__ float float_negate ( float x )
{
  return - x;
}

__device__  float float_recip ( float x )
{
  return 1.0 / x;
}

#endif // CL_CUDA_FLOAT_H_
