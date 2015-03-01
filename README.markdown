# Cl-Cuda

Cl-cuda is a library to use NVIDIA CUDA in Common Lisp programs. It provides not only FFI binding to CUDA driver API but the kernel description language with which users can define CUDA kernel functions in S-expression. The kernel description language also provides facilities to define kernel macros and kernel symbol macros in addition to kernel functions. Cl-cuda's kernel macro and kernel symbol macro offer powerful abstraction that CUDA C itself does not have and provide enormous advantage in resource-limited GPU programming.

Kernel functions defined with the kernel description language can be launched as almost same as ordinal Common Lisp functions except that they must be launched in a CUDA context and followed with grid and block sizes. Kernel functions are compiled and loaded automatically and lazily when they are to be launched for the first time. This process is as following. First, they are compiled into a CUDA C code (.cu file) by cl-cuda. The compiled CUDA C code, then, is compiled into a CUDA kernel module (.ptx file) by NVCC - NVIDIA CUDA Compiler Driver. The obtained kernel module is automatically loaded via CUDA driver API and finally the kernel functions are launched with properly constructed arguments to be passed to CUDA device. Since this process is autonomously managed by the kernel manager, users do not need to handle it for themselves. About the kernel manager, see [Kernel manager](https://github.com/takagi/cl-cuda/blob/master/README.markdown#kernel-manager) section.

Memory management is also one of the most important things in GPU programming. Cl-cuda provides memory block data structure which abstract host memory and device memory. With memory block, users do not need to manage host memory and device memory individually for themselves. It lightens their burden on memory management, prevents bugs and keeps code simple. Besides memory block that provides high level abstraction on host and device memory, cl-cuda also offers low level interfaces to handle CFFI pointers and CUDA device pointers directly. With these primitive interfaces, users can choose to gain more flexible memory control than using memory block if needed.

Cl-cuda is verified on several environments. For detail, see [Verification environments](https://github.com/takagi/cl-cuda/blob/master/README.markdown#verification-environments) section.

## Example

Following code is a part of vector addition example using cl-cuda based on CUDA SDK's "vectorAdd" sample.

You can define `vec-add-kernel` kernel function using `defkernel` macro. In the definition, `aref` is to refer values stored in an array. `set` is to store values into an array. `block-dim-x`, `block-idx-x` and `thread-idx-x` have their counterparts in CUDA C's built-in variables and are used to specify the array index to be operated in each CUDA thread.

Once the kernel function is defined, you can launch it as if it is an ordinal Common Lisp function except that it requires to be in a CUDA context and followed by `:gird-dim` and `:block-dim` keyword parameters which specify the dimensions of grid and block. To keep a CUDA context, you can use `with-cuda` macro which has responsibility on initializing CUDA and managing a CUDA context. `with-memory-blocks` manages memory blocks which abstract host memory area and device memory area, then `sync-memory-block` copies data stored in a memory block between host and device.

For the whole code, please see [examples/vector-add.lisp](https://github.com/takagi/cl-cuda/tree/master/examples/vector-add.lisp).

    (defkernel vec-add-kernel (void ((a float*) (b float*) (c float*) (n int)))
      (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
        (if (< i n)
            (set (aref c i)
                 (+ (aref a i) (aref b i))))))
    
    (defun main ()
      (let* ((dev-id 0)
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

Since cl-cuda is not available in Quicklisp distribution because of its testing policy (see [#514](https://github.com/quicklisp/quicklisp-projects/issues/514) in quicklisp-projects), please use its local-projects feature.

    $ cd ~/quicklisp/local-projects
    $ git clone git://github.com/takagi/cl-cuda.git

Then `(ql:quickload :cl-cuda)` from `REPL` to load it.

## Requirements

Cl-cuda requires following:

* NVIDIA CUDA-enabled GPU
* CUDA Toolkit, CUDA Drivers and CUDA SDK need to be installed

## Verification environments

Cl-cuda is verified to work in following environments:

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
* see issue [#1](https://github.com/takagi/cl-cuda/issues/1#issuecomment-22813518)

#### Environment5 (Thanks to Atabey Kaygun)
* Linux 3.11-2-686-pae SMP Debian 3.11.8-1 (2013-11-13) i686 GNU/Linux
* NVIDIA Corporation GK106 GeForce GTX 660
* CUDA 5.5
* SBCL 1.1.12
* All tests pass, all examples work

## API

Here explain some API commonly used.

### [Macro] with-cuda

    WITH-CUDA (dev-id) &body body

Initializes CUDA and keeps a CUDA context during `body`. `dev-id` are passed to `get-cuda-device` function and the device handler returned is passed to `create-cuda-context` function to create a CUDA context in the expanded form. The results of `get-cuda-device` and `create-cuda-context` functions are bound to `*cuda-device*` and `*cuda-context*` special variables respectively. The kernel manager unloads before `with-cuda` exits.

### [Function] synchronize-context

    SYNCHRONIZE-CONTEXT

Blocks until a CUDA context has completed all preceding requested tasks.

### [Function] alloc-memory-block

    ALLOC-MEMORY-BLOCK type size

Allocates a memory block to hold `size` elements of type `type` and returns it. Actually, linear memory areas are allocated on both host and device memory and a memory block holds pointers to them.

### [Function] free-memory-block

    FREE-MEMORY-BLOCK memory-block

Frees `memory-block` previously allocated by `alloc-memory-block`. Freeing a memory block twice should cause an error.

### [Macro] with-memory-block, with-memory-blocks

    WITH-MEMORY-BLOCK (var type size) &body body
    WITH-MEMORY-BLOCKS ({(var type size)}*) &body body

Binds `var` to a memory block allocated using `alloc-memory-block` applied to the given `type` and `size` during `body`. The memory block is freed using `free-memory-block` when `with-memory-block` exits. `with-memory-blocks` is a plural form of `with-memory-block`.

### [Function] sync-memory-block

    SYNC-MEMORY-BLOCK memory-block direction

Copies stored data between host memory and device memory for `memory-block`. `direction` is either `:host-to-device` or `:device-to-host` which specifies the direction of copying.

### [Accessor] memory-block-aref

    MEMORY-BLOCK-AREF memory-block index

Accesses `memory-block`'s element specified by `index`. Note that the accessed memory area is that on host memory. Use `sync-memory-block` to synchronize stored data between host memory and device memory.

### [Special Variable] \*tmp-path\*

Specifies the temporary directory in which cl-cuda generates files such as `.cu` file and `.ptx` file. The default is `"/tmp/"`.

    (setf *tmp-path* "/path/to/tmp/")

### [Special Variable] \*nvcc-options\*

Specifies additional command-line options to be passed to `nvcc` comand which cl-cuda calls internally. The default is `(list "-arch=sm_11")`.

    (setf *nvcc-options* (list "-arch=sm_20 --verbose"))

### [Special Variable] \*nvcc-binary\*

Specifies the path to `nvcc` command so that cl-cuda can call internally. The default is just `nvcc`.

    (setf *nvcc-binary* "/path/to/nvcc")

### [Special Variable] \*show-messages\*

Specifies whether to let cl-cuda show operational messages or not. The default is `t`.

    (setf *show-messages* nil)

### [Special Variable] \*sdk-not-found\*

Readonly. The value is `nil` if cl-cuda found CUDA SDK when its build, otherwise `t`.

    *sdk-not-found*    ; => nil

## Kernel Description Language

### Types

not documented yet.

### IF statement

    IF test-form then-form [else-form]

`if` allows the execution of a form to be dependent on a single `test-form`. First `test-form` is evaluated. If the result is `true`, then `then-form` is selected; otherwise `else-form` is selected. Whichever form is selected is then evaluated. If `else-form` is not provided, does nothing when `else-form` is selected.

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

    LET ({(var init-form)}*) statement*

`let` declares new variable bindings and set corresponding `init-form`s to them and execute a series of `statement`s that use these bindings. `let` performs the bindings in parallel. For sequentially, use `let*` kernel macro instead.

Example:

    (let ((i 0))
      (return i))

Compiled:

    {
      int i = 0;
      return i;
    }

### SYMBOL-MACROLET statement

    SYMBOL-MACROLET ({(symbol expansion)}*) statement*

`symbol-macrolet` establishes symbol expansion rules in the variable environment and execute a series of `statement`s that use these rules. In cl-cuda's compilation process, the symbol macros found in a form are replaces by corresponding `expansion`s.

Example:

    (symbol-macrolet ((x 1.0))
      (return x))

Compiled:

    {
      return 1.0;
    }

### DO statement

    DO ({(var init-form step-form)}*) (test-form) statement*

`do` iterates over a group of `statement`s while `test-form` holds. `do` accepts an arbitrary number of iteration `var`s and their initial values are supplied by `init-form`s. `step-form`s supply how the `var`s should be updated on succeeding iterations through the loop.

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

    WITH-SHARED-MEMORY ({(var type size*)}*) statement*

`with-shared-memory` declares new variable bindings on shared memory by adding `__shared__` variable specifiers. It allows to declare array variables if dimensions are provided. A series of `statement`s are executed with these bindings.

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

    SET reference expression

`set` provides simple variable assignment. It accepts one of variable, structure and array references as `reference`.

Example:

    (set x 1.0)
    (set (float4-x y 1.0)
    (set (aref z 0) 1.0)

Compiled:

    x = 1.0;
    y.x = 1.0;
    z[0] = 1.0;

### PROGN statement

    PROGN statement*

`progn` evaluates `statement`s, in the order in which they are given.

Example:

    (progn
      (do-some-statements)
      (do-more-statements))

Compiled:

    do_some_statements();
    do_more_statements();

### RETURN statement

    RETURN [return-form]

`return` returns control, with `return-form` if supplied, from a kernel function.

Example:

    (return 0)

Compiled:

    return 0;

## Architecture

The following figure illustrates cl-cuda's overall architecture.

                       +---------------------------------+-----------+-----------+
                       | defkernel                       | memory    | context   |
           cl-cuda.api +---------------------------------+           |           |
                       | kernel-manager                  |           |           |
                       +---------------------------------+-----------+-----------+
                       +----------------------------+----------------------------+
          cl-cuda.lang | Kernel description lang.   | the Compiler               |
                       +----------------------------+----------------------------+
                       +---------------------------------------------------------+
    cl-cuda.driver-api | driver-api                                              |
                       +---------------------------------------------------------+
                       +---------------------------------------------------------+
                  CUDA | CUDA driver API                                         |
                       +---------------------------------------------------------+

Cl-cuda consists of three subpackages: `api`, `lang` and `driver-api`.

`driver-api` subpackage is a FFI binding to CUDA driver API. `api` subpackage invokes CUDA driver API via this binding internally.

`lang` subpackage provides the kernel description language. It provides the language's syntax, type, built-in functions and the compiler to CUDA C. `api` subpackage calls this compiler.

`api` subpackage provides API for cl-cuda users. It further consists of `context`, `memory`, `kernel-manager` and `defkernel` subpackages. `context` subpackage has responsibility on initializing CUDA and managing CUDA contexts. `memory` subpackage offers memory management, providing high level API for memory block data structure and low level API for handling host memory and device memory directly. `kernel-manager` subpackage manages the entire process from compiling the kernel description language to loading/unloading obtained kernel module autonomously. Since it is wrapped by `defkernel` subpackage which provides the interface to define kernel functions, cl-cuda's users usually do not need to use it for themselves.

## Kernel manager

The kernel manager is a module which manages defining kernel functions, compiling them into a CUDA kernel module, loading it and unloading it. I show you its work as a finite state machine here.

To begin with, the kernel manager has four states.

    I   initial state
    II  compiled state
    III module-loaded state
    IV  function-loaded state

The initial state is its entry point. The compiled state is a state where kernel functions defined with the kernel descrpition language have been compiled into a CUDA kernel module (.ptx file). The obtained kernel module has been loaded in the module-loaded state. In the function-loaded state, each kernel function in the kernel module has been loaded.

Following illustrates the kernel manager's state transfer.

    　    compile-module        load-module            load-function
    　  =================>    =================>     =================>
    　I                    II                    III                    IV
    　  <=================    <=================
    　    define-function     <========================================
    　    define-macro          unload
    　    define-symbol-macro

`kernel-manager-compile-module` function compiles defined kernel functions into a CUDA kernel module. `kernel-manager-load-module` function loads the obtained kernel module. `kernel-manager-load-function` function loads each kernel function in the kernel module.

In the module-loaded state and function-loaded state, `kernel-manager-unload` function unloads the kernel module and turn the kernel manager's state back to the compiled state. `kernel-manager-define-function`, `kernel-manager-define-macro` and `kernel-manager-define-symbol-macro` functions, which are wrapped as `defkernel`, `defkernelmacro` and `defkernel-symbol-macro` macros respectively, change its state back into the initial state and make it require compilation again.

The kernel manager is stored in `*kernel-manager*` special variable when cl-cuda is loaded and keeps alive during the Common Lisp process. Usually, you do not need to manage it explicitly.

## How cl-cuda works when CUDA SDK is not installed

This section is for cl-cuda users who develop an application or a library which has alternative sub system other than cl-cuda and may run on environments CUDA SDK is not installed.

**Compile and load time**
Cl-cuda is compiled and loaded without causing any conditions on environments CUDA SDK is not installed. Since cl-cuda API 's symbols are interned, user programs can use them normally.

**Run time**
At the time cl-cuda's API is called, an error that tells CUDA SDK is not found should occur. With `*sdk-not-found*` special variable, user programs can get if cl-cuda has found CUDA SDK or not.

How cl-cuda determines CUDA SDK is installed or not is that if it has successfully loaded `libuda` dynamic library with `cffi:user-foreign-library` function.

## Streams

The low level interface works with multiple streams. With the async stuff it's possible to overlap copy and computation with two streams. Cl-cuda provides `*cuda-stream*` special variable, to which bound stream is used in kernel function calls.

The following is for working with streams in [mgl-mat](https://github.com/melisgl/mgl-mat):

    (defmacro with-cuda-stream ((stream) &body body)
      (alexandria:with-gensyms (stream-pointer)
        `(cffi:with-foreign-objects
             ((,stream-pointer 'cl-cuda.driver-api:cu-stream))
           (cl-cuda.driver-api:cu-stream-create ,stream-pointer 0)
           (let ((,stream (cffi:mem-ref ,stream-pointer
                                        'cl-cuda.driver-api:cu-stream)))
             (unwind-protect
                  (locally ,@body)
               (cl-cuda.driver-api:cu-stream-destroy ,stream))))))

then, call a kernel function with binding a stream to `*cuda-stream*`:

    (with-cuda-stream (*cuda-stream*)
      (call-kernel-function))

## Author

* Masayuki Takagi (kamonama@gmail.com)

## Copyright

Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)

## License

Licensed under the LLGPL License.
