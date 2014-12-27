/**
   This file is a part of cl-cuda project.
   Copyright (c) 2013 Masayuki Takagi (kamonama@gmail.com)
 */

#ifndef CL_CUDA_CURAND_H_
#define CL_CUDA_CURAND_H_

#include <curand_kernel.h>

__device__ void curand_init_xorwow ( int seed, int sequence, int offset,
                                     curandStateXORWOW_t *state )
{
    curand_init ( seed, sequence, offset, state);
}

__device__ float curand_uniform_float_xorwow ( curandStateXORWOW_t *state )
{
    return curand_uniform ( state );
}

__device__ double curand_uniform_double_xorwow ( curandStateXORWOW_t *state )
{
    return curand_uniform_double ( state );
}

__device__ float curand_normal_float_xorwow ( curandStateXORWOW_t *state )
{
    return curand_normal ( state );
}

__device__ double curand_normal_double_xorwow ( curandStateXORWOW_t *state )
{
    return curand_normal_double ( state );
}

#endif // CL_CUDA_CURAND_H_
