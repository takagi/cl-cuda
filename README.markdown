# Cl-Cuda

Cl-cuda is a library to use Nvidia CUDA in Common Lisp programs. You can write CUDA kernel functions using the cl-cuda kernel description language which has Common-Lisp-like syntax.

Cl-cuda is in very early stage of development. Any feedbacks are welcome.

## Example

Following is a part of vector addition example using cl-cuda which is based on the CUDA SDK's "vectorAdd" sample.

Kernel functions are simply written with **defkernel** macro and the cl-cuda kernel description language which has Common Lisp-like syntax.

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

I will write about installation later.

## Rquirements

Since cl-cuda now uses some features that depend on SBCL, you will need some modification if you use it on any Common Lisp implementations other than SBCL. I will fix this later.

## Author

* Masayuki Takagi (kamonama@gmail.com)

## Copyright

Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)

# License

Licensed under the LLGPL License.

