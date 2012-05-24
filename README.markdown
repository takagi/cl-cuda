# Cl-Cuda

Cl-cuda is a library to use Nvidia CUDA in Common Lisp programs. You can write CUDA kernel functions using the cl-cuda kernel description language which has Common Lisp-like syntax.

Cl-cuda is in very early stage of development. Any feedbacks are welcome.

## Example

Following is a part of vector addition example using cl-cuda which is based on the CUDA SDK's "vectorAdd" sample.

Kernel functions are simply written with `defkernel` macro and the cl-cuda kernel description language which has Common Lisp-like syntax.

Once kernel functions are defined, they can be launched as if ordinal Common Lisp functions except that they are followed by :grid-dim and :block-dim keyword parameters which provide the dimensions of the grid and block.

For the whole code, please see examples/vector-add.lisp.

    (defkernel vec-add-kernel (void ((a float*) (b float*) (c float*) (n int)))
      (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
        (if (< i n)
            (set (aref c i)
                 (+ (aref a i) (aref b i))))))
    
    (let ((dev-id 0))
      (with-cuda-context (dev-id)
        ...
        (vec-add-kernel d-a d-b d-c n
                        :grid-dim (list blocks-per-grid 1 1)
                        :block-dim (list threads-per-block 1 1))
        ...))

## Usage

I will write some usage later. For now, please see the examples directory.

## Installation

Since cl-cuda is not registered on Quicklisp yet, please

    git clone git clone git://github.com/takagi/cl-cuda.git

to install it.

Before using cl-cuda, you must specify where **libcuda** dynamic library is and where nvcc compiler is. Please change the related part of src/cl-cuda.lisp. I will make better way to specify them later.

I will write more about installation later.

## Rquirements

* NVIDIA CUDA-enabled GPU
* CUDA Toolkit, CUDA Drivers and CUDA SDK need to be installed
* SBCL Common Lisp compiler, because cl-cuda uses some sbcl extensions to run nvcc compiler externally. I will fix it later to make it possible to be used on other Common Lisp implementations. For now, if you want to use cl-cuda on those implementations other than SBCL, you can rewrite the related part of src/cl-cuda.lisp to suit your environment. It is only a few lines.

## Author

* Masayuki Takagi (kamonama@gmail.com)

## Copyright

Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)

# License

Licensed under the LLGPL License.

