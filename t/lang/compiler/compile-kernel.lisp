#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.lang.compiler.compile-kernel
  (:use :cl :cl-test-more
        :cl-cuda.lang.type
        :cl-cuda.lang.kernel
        :cl-cuda.lang.compiler.compile-kernel))
(in-package :cl-cuda-test.lang.compiler.compile-kernel)

(plan nil)


;;;
;;; test COMPILE-KERNEL funcition
;;;

(diag "COMPILE-KERNEL")

(let ((kernel (make-kernel)))
  (kernel-define-function kernel 'foo 'void '((x int*))
                                 '((set (aref x 0) (bar 1))
                                   (return)))
  (kernel-define-function kernel 'bar 'int '((x int)) '((return x)))
  (kernel-define-function kernel 'baz 'void '() '((return)))
  (is (compile-kernel kernel)
      "#include \"int.h\"
#include \"float.h\"
#include \"float3.h\"
#include \"float4.h\"
#include \"double.h\"
#include \"double3.h\"
#include \"double4.h\"
#include \"curand.h\"


/**
 *  Kernel function prototypes
 */

extern \"C\" __global__ void cl_cuda_test_lang_compiler_compile_kernel_baz();
extern \"C\" __device__ int cl_cuda_test_lang_compiler_compile_kernel_bar( int x );
extern \"C\" __global__ void cl_cuda_test_lang_compiler_compile_kernel_foo( int* x );


/**
 *  Kernel function definitions
 */

__global__ void cl_cuda_test_lang_compiler_compile_kernel_baz()
{
  return;
}

__device__ int cl_cuda_test_lang_compiler_compile_kernel_bar( int x )
{
  return x;
}

__global__ void cl_cuda_test_lang_compiler_compile_kernel_foo( int* x )
{
  x[0] = cl_cuda_test_lang_compiler_compile_kernel_bar( 1 );
  return;
}
"
      "basic case 1"))


(finalize)
