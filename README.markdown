# Cl-Cuda

Cl-cuda is a library to use NVIDIA CUDA in Common Lisp programs. You can write CUDA kernel functions using the cl-cuda kernel description language which has Common Lisp-like syntax.

Cl-cuda is in very early stage of development. Any feedbacks are welcome.

## Example

Following is a part of vector addition example using cl-cuda based on the CUDA SDK's "vectorAdd" sample.

Kernel functions are simply written with `defkernel` macro and the cl-cuda kernel description language which has Common Lisp-like syntax.

Once kernel functions are defined, they can be launched as if ordinal Common Lisp functions except that they are followed by `:grid-dim` and `:block-dim` keyword parameters which provide the dimensions of grid and block.

For the whole code, please see examples/vector-add.lisp.

    (defkernel vec-add-kernel (void ((a float*) (b float*) (c float*) (n int)))
      (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
        (if (< i n)
            (set (aref c i)
                 (+ (aref a i) (aref b i))))))
    
    (defun main ()
      (let ((dev-id 0)
            (n 1024)
            (threads-per-block 256)
            (blocks-per-grid (/ n threads-per-block)))
        (with-cuda (dev-id)
          (with-memory-blocks ((a 'float n)
                               (b 'float n)
                               (c 'float n))
            (random-init a n)
            (random-init b n)
            (sync-memory-block a :host-to-device)
            (sync-memory-block b :host-to-device)
            (vec-add-kernel a b c n
                            :grid-dim  (list blocks-per-grid 1 1)
                            :block-dim (list threads-per-block 1 1))
            (sync-memory-block c :device-to-host)
            (verify-result a b c n)))))

## Installation

Since cl-cuda is not available in Quicklisp distribution, please use Quicklisp's local-projects feature.

    $ cd ~/quicklisp/local-projects
    $ git clone git://github.com/takagi/cl-cuda.git

Then `(ql:quickload :cl-cuda)` from `REPL` to load it.

## Requirements

* NVIDIA CUDA-enabled GPU
* CUDA Toolkit, CUDA Drivers and CUDA SDK need to be installed

## Verification environment

#### Environment 1
* Mac OS X 10.6.8 (MacBookPro)
* GeForce 9400M
* CUDA 4
* SBCL 1.0.55 32-bit
* All tests pass, all examples work

#### Environment2
* Amazon Linux x86_64 (Amazon EC2)
* Tesla M2050
* CUDA 4
* SBCL 1.1.7 64-bit
* All tests pass, all examples which are verified work (others not tried yet)
* `(setf *nvcc-options* (list "-arch=sm_20" "-m32"))` needed

#### Environment3 (Thanks to Viktor Cerovski)
* Linux 3.5.0-32-generic Ubuntu SMP x86_64
* GeFroce 9800 GT
* CUDA 5
* SBCL 1.1.7 64-bit
* All tests pass, all examples work

#### Environment4 (Thanks to wvxvw)
* Fedra18 x86_64
* GeForce GTX 560M
* CUDA 5.5
* SBCL 1.1.2-1.fc18
* `vector-add` example works (didn't try the rest yet)

Further information: 
* `(setf *nvcc-options* (list "-arch=sm_20" "-m32"))` needed
* using video drivers from `rpmfusion` instead of the ones in `cuda` package
* see https://github.com/takagi/cl-cuda/issues/1#issuecomment-22813518

#### Environment5 (Thanks to Atabey Kaygun)
* Linux 3.11-2-686-pae SMP Debian 3.11.8-1 (2013-11-13) i686 GNU/Linux
* NVIDIA Corporation GK106 [GeForce GTX 660] (rev a1)
* CUDA 5.5
* SBCL 1.1.12
* All tests pass, all examples work

## API

### [Function] init-cuda-context

    init-cuda-context dev-id &key (interop nil)

Initializes the CUDA driver API, creates a new CUDA context and associates it with the calling thread. The `dev-id` parameter specifies a device number to get handle for. If the `interop` parameter is `nil`, an usual CUDA context is created. Otherwise, a CUDA context is created for OpenGL interoperability.

If initialization or context creation will fail, `init-cuda-context` will be cancelled with calling `release-cuda-context`.

### [Function] release-cuda-context

    release-cuda-context

Unloads a kernel module, destroys a CUDA context and releases all related resources. If a kernel module is not loaded, `release-cuda-context` raises no error, just does nothing. Similarly, if a CUDA context is not created, it just does nothing about it.

### [Macro] with-cuda-context

    with-cuda-context (dev-id &key (interop nil)) &body body

Keeps a CUDA context during `body`. The `dev-id` and `interop` parameters are passed to `init-cuda-context` function which appears in its expansion form.

### [Function] synchronize-context

    synchronize-context

Blocks until the device has completed all preceding requested tasks.

### [Function] alloc-memory-block

    alloc-memory-block type n &key (interop nil) => memory block

Allocates a memory block to hold `n` elements of type `type` and returns it. Actually, linear memory areas are allocated on both device and host memory respectively, and a memory block holds pointers to the areas to abstract them.

If the `interop` parameter is nil, linear memory areas are allocated to be used for an usual CUDA context. Otherwise, they are allocated to be used for an CUDA context under OpenGL interoperability.

### [Function] free-memory-block

    free-memory-block block

Frees `block` previously allocated by `alloc-memory-block`. Freeing a given memory block twice does nothing.

### [Macro] with-memory-block, with-memory-blocks

    with-memory-block (var type size &key (interop nil)) &body body
    
    with-memory-blocks (bindings) &body body
    bindings ::= {(var type size &key (interop nil))}*

Bind `var` to a memory block allocated using `alloc-memory-block` applied to given `type`, `size` and `interop` parameters during `body`. `with-memory-blocks` is a plural form of `with-memory-block`.

### [Accessor] mem-aref

    mem-aref block index => value

Accesses the memory block `block` element specified by the `index`. Since the accessed memory area via `mem-aref` is that on host memory, use `memcpy-host-to-device` and `memcpy-device-to-host` functions to synchronize stored data on host and device memory areas.

### [Function] memcpy-host-to-device

    memcpy-host-to-device &rest blocks

Copies from host memory to device memory for given memory blocks `blocks` which abstract the memory areas.

### [Function] memcpy-device-to-host

    memcpy-device-to-host &rest blocks

Copies from device memory to host memory for given memory blocks `blocks` which abstract the memory areas.

### [Special Variable] \*nvcc-options\*

Specifies additional command-line options to be pass to the NVIDIA CUDA Compiler which cl-cuda calls internally.

Default: `(list "-arch=sm_11")`

    (setf *nvcc-options* (list "-arch=sm_20 --verbose"))

### [Special Variable] \*tmp-path\*

Specifies the temporary directory in which cl-cuda generates files such as .cu file and .ptx file to compile kernel module.

Default: `"/tmp/"`

    (setf *tmp-path* "/path/to/tmp/")

### [Special Variable] \*nvcc-binary\*

Specifies the path to the NVIDIA CUDA Compiler so that cl-cuda can call it internally.

Default: `nvcc`

    (setf *nvcc-binary* "/path/to/nvcc")

### [Special Variable] \*show-messages\*

Specifies whether to let cl-cuda show operational messages or not.

Default: `t`

    (setf *show-messages* t)

## Kernel Definition Language

### DEFKERNEL macro

### launching kernel functions

### Types

### IF statement

Syntax:

    IF test-form then-form [else-form]

Example:

    (if (= a 0)
        (return 0)
        (return 1))

Compiled:

    if (a == 0) {
      return 0;
    } else {
      return 1;
    }

### LET statement

Syntax:

    LET ({(var init-form)}*) statement*

Example:

    (let ((i 0))
      (return i))

Compiled:

    {
      int i = 0;
      return i;
    }

### DO statement

Syntax:

    DO ({(var init-form step-form)}*) (test-form) statement*

Example:

    (do ((a 0 (+ a 1))
         (b 0 (+ b 1)))
        ((> a 15))
      (do-some-statement))

Compiled:

    for ( int a = 0, int b = 0; ! (a > 15); a = a + 1, b = b + 1 )
    {
      do_some_statement();
    }

### WITH-SHARED-MEMORY statement

Syntax:

    WITH-SHARED-MEMORY ({(var type size*)}*) statement*

Example:

    (with-shared-memory ((a int 16)
                         (b float 16 16))
      (return))

Compiled:

    {
      __shared__ int a[16];
      __shared__ float b[16][16];
      return;
    }

### SET statement

### PROGN statement

Syntax:

    PROGN statement*

Example:

    (progn
      (do-some-statements)
      (do-more-statements))

Compiled:

    do_some_statements();
    do_more_statements();

### RETURN statement

Syntax:

    RETURN [return-form]

Example:

    (return 0)

Compiled:

    return 0;

## Author

* Masayuki Takagi (kamonama@gmail.com)

## Copyright

Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)

## License

Licensed under the LLGPL License.
