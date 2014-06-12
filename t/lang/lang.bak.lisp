#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-test.lang)

(plan nil)


;;;
;;; test kernel definition
;;;

(diag "test kernel definition")

;; test making empty kernel definition
(is (cl-cuda.lang::empty-kernel-definition) '(nil nil))

;; test adding function to kernel definition
(let ((def (cl-cuda.lang::add-function-to-kernel-definition 'foo 'void '() '((return))
             (cl-cuda.lang::empty-kernel-definition))))
  (is (cl-cuda.lang::kernel-definition-function-exists-p 'foo def) t))

;; test removing function from kernel definition
(let ((def (cl-cuda.lang::remove-function-from-kernel-definition 'foo
             (cl-cuda.lang::add-function-to-kernel-definition 'foo 'void '() '((return))
               (cl-cuda.lang::empty-kernel-definition)))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; can not remove function which does not exist in kernel definition
(is-error (cl-cuda.lang::remove-function-from-kernel-definition 'foo
            (cl-cuda.lang::empty-kernel-definition)) simple-error)

;; kernel definition does not shadow its elements, just overwrites
(let ((def (cl-cuda.lang::remove-function-from-kernel-definition 'foo
             (cl-cuda.lang::add-function-to-kernel-definition 'foo 'void '() '((return))
               (cl-cuda.lang::add-function-to-kernel-definition 'foo 'int '() '((return 1))
                 (cl-cuda.lang::empty-kernel-definition))))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; test adding macro to kernel definition
(let ((def (cl-cuda.lang::add-macro-to-kernel-definition 'foo '(x) '(`(expanded ,x))
                                                    (lambda (args) (destructuring-bind (x) args `(expanded ,x)))
             (cl-cuda.lang::empty-kernel-definition))))
  (is (cl-cuda.lang::kernel-definition-macro-exists-p 'foo def) t))

;; test removing macro from kernel definition
(let ((def (cl-cuda.lang::remove-macro-from-kernel-definition 'foo
             (cl-cuda.lang::add-macro-to-kernel-definition 'foo '() '(`(expanded ,x))
                                                      (lambda (args) (destructuring-bind (x) args `(expanded ,x)))
               (cl-cuda.lang::empty-kernel-definition)))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; can not remove macro which does not exist in kernel definition
(is-error (cl-cuda.lang::remove-macro-from-kernel-definition 'foo
            (cl-cuda.lang::empty-kernel-definition)) simple-error)

;; kernel definition does not shadow its elements, just overwrites
(let ((def (cl-cuda.lang::remove-macro-from-kernel-definition 'foo
             (cl-cuda.lang::add-macro-to-kernel-definition 'foo '(x) '(`(expanded ,x))
                                                      (lambda (args) (destructuring-bind (x) args `(expanded ,x)))
               (cl-cuda.lang::add-macro-to-kernel-definition 'foo '() '('(return))
                                                        (lambda (args) (destructuring-bind () args '(return)))
                 (cl-cuda.lang::empty-kernel-definition))))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; test adding constant to kernel definition
(let ((def (cl-cuda.lang::add-constant-to-kernel-definition 'x 'float 1.0
             (cl-cuda.lang::empty-kernel-definition))))
  (is (cl-cuda.lang::kernel-definition-constant-exists-p 'x def) t))

;; test removing constant from kernel definition
(let ((def (cl-cuda.lang::remove-constant-from-kernel-definition 'x
             (cl-cuda.lang::add-constant-to-kernel-definition 'x 'float 1.0
               (cl-cuda.lang::empty-kernel-definition)))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; can not remove constant which does not exist in kernel definition
(is-error (cl-cuda.lang::remove-constant-from-kernel-definition 'x
            (cl-cuda.lang::empty-kernel-definition)) simple-error)

;; kernel definition does not shadow its elements, just overwrites
(let ((def (cl-cuda.lang::remove-constant-from-kernel-definition 'x
             (cl-cuda.lang::add-constant-to-kernel-definition 'x 'int 1
               (cl-cuda.lang::add-constant-to-kernel-definition 'x 'float 1.0
                 (cl-cuda.lang::empty-kernel-definition))))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; test adding symbol macro to kernel definition
(let ((def (cl-cuda.lang::add-symbol-macro-to-kernel-definition 'x 1.0
             (cl-cuda.lang::empty-kernel-definition))))
  (is (cl-cuda.lang::kernel-definition-symbol-macro-exists-p 'x def) t))

;; test removing symbol macro from kernel definition
(let ((def (cl-cuda.lang::remove-symbol-macro-from-kernel-definition 'x
             (cl-cuda.lang::add-symbol-macro-to-kernel-definition 'x 1.0
               (cl-cuda.lang::empty-kernel-definition)))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; can not remove symbol macro which does not exist in kernel definition
(is-error (cl-cuda.lang::remove-symbol-macro-from-kernel-definition 'x
            (cl-cuda.lang::empty-kernel-definition)) simple-error)

;; kernel definition does not shadow its elements, just overwrites
(let ((def (cl-cuda.lang::remove-symbol-macro-from-kernel-definition 'x
             (cl-cuda.lang::add-symbol-macro-to-kernel-definition 'x 1
               (cl-cuda.lang::add-symbol-macro-to-kernel-definition 'x 1.0
                 (cl-cuda.lang::empty-kernel-definition))))))
  (is def (cl-cuda.lang::empty-kernel-definition)))

;; test kernel definition
(cl-cuda.lang::with-kernel-definition (def ((f :function int ((x int)) ((return x)))
                                       (g :macro (x) (`(expanded ,x)))
                                       (x :constant float 1.0)
                                       (y :symbol-macro 1.0)))
  ;; test predicates
  (is       (cl-cuda.lang::kernel-definition-function-exists-p 'f def) t)
  (is       (cl-cuda.lang::kernel-definition-function-exists-p 'g def) nil)
  (is       (cl-cuda.lang::kernel-definition-macro-exists-p 'g def) t)
  (is       (cl-cuda.lang::kernel-definition-macro-exists-p 'f def) nil)
  (is       (cl-cuda.lang::kernel-definition-constant-exists-p 'x def) t)
  (is       (cl-cuda.lang::kernel-definition-constant-exists-p 'f def) nil)
  (is       (cl-cuda.lang::kernel-definition-symbol-macro-exists-p 'y def) t)
  (is       (cl-cuda.lang::kernel-definition-symbol-macro-exists-p 'f def) nil)
  ;; test selectors
  (is       (cl-cuda.lang::kernel-definition-function-name 'f def) 'f)
  (is-error (cl-cuda.lang::kernel-definition-function-name 'g def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-function-c-name 'f def) "cl_cuda_test_lang_f")
  (is-error (cl-cuda.lang::kernel-definition-function-c-name 'g def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-function-names def) '(f))
  (is       (cl-cuda.lang::kernel-definition-function-return-type 'f def) 'int)
  (is-error (cl-cuda.lang::kernel-definition-function-return-type 'g def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-function-arguments 'f def) '((x int)))
  (is-error (cl-cuda.lang::kernel-definition-function-arguments 'g def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-function-argument-types 'f def) '(int))
  (is-error (cl-cuda.lang::kernel-definition-function-argument-types 'g def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-function-body 'f def) '((return x )))
  (is-error (cl-cuda.lang::kernel-definition-function-body 'g def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-macro-name 'g def) 'g)
  (is-error (cl-cuda.lang::kernel-definition-macro-name 'f def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-macro-names def) '(g))
  (is       (cl-cuda.lang::kernel-definition-macro-arguments 'g def) '(x))
  (is-error (cl-cuda.lang::kernel-definition-macro-arguments 'f def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-macro-body 'g def) '(`(expanded ,x)))
  (is-error (cl-cuda.lang::kernel-definition-macro-body 'f def) simple-error)
  (is       (funcall (cl-cuda.lang::kernel-definition-macro-expander 'g def) '(x)) '(expanded x))
  (is-error (funcall (cl-cuda.lang::kernel-definition-macro-expander 'f def) '(x)) simple-error)
  (is       (cl-cuda.lang::kernel-definition-constant-name 'x def) 'x)
  (is-error (cl-cuda.lang::kernel-definition-constant-name 'f def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-constant-names def) '(x))
  (is       (cl-cuda.lang::kernel-definition-constant-type 'x def) 'float)
  (is-error (cl-cuda.lang::kernel-definition-constant-type 'f def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-constant-expression 'x def) 1.0)
  (is-error (cl-cuda.lang::kernel-definition-constant-expression 'f def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-symbol-macro-name 'y def) 'y)
  (is-error (cl-cuda.lang::kernel-definition-symbol-macro-name 'f def) simple-error)
  (is       (cl-cuda.lang::kernel-definition-symbol-macro-names def) '(y))
  (is       (cl-cuda.lang::kernel-definition-symbol-macro-expansion 'y def) 1.0)
  (is-error (cl-cuda.lang::kernel-definition-symbol-macro-expansion 'f def) simple-error))

;; kernel definition does not shadow its elements, just overwrites
(cl-cuda.lang::with-kernel-definition (def ((f :function void () ((return)))
                                       (f :function int ((x int)) ((return x)))
                                       (g :macro () ('(return)))
                                       (g :macro (x) (`(expanded ,x)))
                                       (h :function void () ((return)))
                                       (h :macro (x) (`(expanded ,x)))
                                       (x :constant float 1.0)
                                       (x :constant float 2.0)
                                       (y :symbol-macro 1.0)
                                       (y :symbol-macro 2.0)))
  (is (cl-cuda.lang::kernel-definition-function-return-type 'f def) 'int)
  (is (cl-cuda.lang::kernel-definition-macro-arguments 'g def) '(x))
  (is (cl-cuda.lang::kernel-definition-function-exists-p 'h def) nil)
  (is (cl-cuda.lang::kernel-definition-macro-exists-p 'h def) t)
  (is (cl-cuda.lang::kernel-definition-constant-expression 'x def) 2.0)
  (is (cl-cuda.lang::kernel-definition-symbol-macro-expansion 'y def) 2.0))

;; kernel definition can accept element depending on others
(cl-cuda.lang::with-kernel-definition (def ((x :constant float 1.0)
                                       (y :constant float (* x 2.0))))
  (is (cl-cuda.lang::kernel-definition-constant-exists-p 'x def) t)
  (is (cl-cuda.lang::kernel-definition-constant-exists-p 'y def) t))

;; kernel definition does not accept variables
(is-error (cl-cuda.lang::with-kernel-definition (def ((x :variable int 1))) def) simple-error)


;;;
;;; test compile-kernel-definition
;;;

(diag "test compile-kernel-definition")

(let ((def (cl-cuda.lang::add-function-to-kernel-definition 'foo 'void '() '((return))
             (cl-cuda.lang::empty-kernel-definition)))
      (c-code (cl-cuda.lang::unlines "#include \"int.h\""
                                "#include \"float.h\""
                                "#include \"float3.h\""
                                "#include \"float4.h\""
                                "#include \"double.h\""
                                "#include \"double3.h\""
                                "#include \"double4.h\""
                                "#include \"curand.h\""
                                ""
                                "extern \"C\" __global__ void cl_cuda_test_lang_foo ();"
                                ""
                                "__global__ void cl_cuda_test_lang_foo ()"
                                "{"
                                "  return;"
                                "}"
                                "")))
  (is (cl-cuda.lang::compile-kernel-definition def) c-code))


;;;
;;; test compile-kernel-function-prototype
;;;

(diag "test compile-kernel-function-prototype")

(let ((def (cl-cuda.lang::add-function-to-kernel-definition 'foo 'void '() '((return))
             (cl-cuda.lang::empty-kernel-definition)))
      (c-code (cl-cuda.lang::unlines "extern \"C\" __global__ void cl_cuda_test_lang_foo ();")))
  (is (cl-cuda.lang::compile-kernel-function-prototype 'foo def) c-code))


;;;
;;; test compile-kernel-function
;;;

(diag "test compile-kernel-function")

(let ((def (cl-cuda.lang::add-function-to-kernel-definition 'foo 'void '() '((return))
             (cl-cuda.lang::empty-kernel-definition)))
      (c-code (cl-cuda.lang::unlines "__global__ void cl_cuda_test_lang_foo ()"
                                "{"
                                "  return;"
                                "}"
                                "")))
  (is (cl-cuda.lang::compile-kernel-function 'foo def) c-code))


;;;
;;; test compile-function-specifier (not implemented)
;;;



;;;
;;; test compile-type (not implemented)
;;;



;;;
;;; test compile-identifier
;;;

(diag "test compile-identifier")

;; test compile-identifier
(is (cl-cuda.lang::compile-identifier 'x             )  "x"             )
(is (cl-cuda.lang::compile-identifier 'vec-add-kernel)  "vec_add_kernel")
(is (cl-cuda.lang::compile-identifier 'vec.add.kernel)  "vec_add_kernel")
(is (cl-cuda.lang::compile-identifier '%vec-add-kernel) "_vec_add_kernel")
(is (cl-cuda.lang::compile-identifier 'VecAdd_kernel )  "vecadd_kernel" )


;;;
;;; test compile-if
;;;

(diag "test compile-if")

(let ((lisp-code '(if t
                      (return)
                      (return)))
      (c-code (cl-cuda.lang::unlines "if (true) {"
                                "  return;"
                                "} else {"
                                "  return;"
                                "}")))
  (is (cl-cuda.lang::compile-if lisp-code nil nil) c-code))

(let ((lisp-code '(if t
                      (progn
                        (return 0)
                        (return 0))))
      (c-code (cl-cuda.lang::unlines "if (true) {"
                                "  return 0;"
                                "  return 0;"
                                "}")))
  (is (cl-cuda.lang::compile-if lisp-code nil nil) c-code))

(let ((lisp-code '(if 1
                      (return)
                      (return))))
  (is-error (cl-cuda.lang::compile-if lisp-code nil nil) simple-error))


;;;
;;; test compile-let  
;;;

(diag "test compile-let")

(let ((lisp-code '(let ((i 0))
                    (return)
                    (return)))
      (c-code (cl-cuda.lang::unlines "{"
                                "  int i = 0;"
                                "  return;"
                                "  return;"
                                "}")))
  (is (cl-cuda.lang::compile-let lisp-code nil nil) c-code))

(is-error (cl-cuda.lang::compile-let '(let (i) (return)) nil nil) simple-error)
(is-error (cl-cuda.lang::compile-let '(let ((i)) (return)) nil nil) simple-error)
(is-error (cl-cuda.lang::compile-let '(let ((x 1) (y x)) (return y)) nil nil) simple-error)

(let ((lisp-code '(let* ((x 1)
                         (y x))
                   (return y)))
      (c-code (cl-cuda.lang::unlines "{"
                                "  int x = 1;"
                                "  int y = x;"
                                "  return y;"
                                "}")))
  (is (cl-cuda.lang::compile-let* lisp-code nil nil) c-code))


;;;
;;; test compile-symbol-macrolet
;;;

(diag "test compile-symbol-macrolet")

(is (cl-cuda.lang::symbol-macrolet-p '(symbol-macrolet ((x 'expanded-x)) (return))) t)
(is (cl-cuda.lang::symbol-macrolet-p '(symbol-macrolet ((x 'expanded-x)) (do-something) (return))) t)
(is (cl-cuda.lang::symbol-macrolet-p '(symbol-macrolet ((x 'expanded-x)))) t)

(let ((lisp-code '(symbol-macrolet ((x expanded-x))
                    (let ((expanded-x 1.0))
                      (return x))))
      (c-code (cl-cuda.lang::unlines "{"
                                "  float expanded_x = 1.0;"
                                "  return expanded_x;"
                                "}")))
  (is (cl-cuda.lang::compile-symbol-macrolet lisp-code nil nil) c-code))

(is-error (cl-cuda.lang::compile-symbol-macrolet '(symbol-macrolet (x) (return)) nil nil) simple-error)
(is-error (cl-cuda.lang::compile-symbol-macrolet '(symbol-macrolet ((x)) (return)) nil nil) simple-error)


;;;
;;; test compile-do
;;;

(diag "test compile-do")

;; test do selectors
(let* ((code '(do ((a 0 (+ a 1))
                   (b 0 (+ b 1)))
                  ((> a 15))
                (return)))
       (binding (first (cl-cuda.lang::do-bindings code))))
  (cl-cuda.lang::with-variable-environment (var-env ((a :variable int) (b :variable int)))
    (is (cl-cuda.lang::do-p code)                    t)
    (is (cl-cuda.lang::do-bindings code)             '((a 0 (+ a 1))
                                                  (b 0 (+ b 1))))
    (is (cl-cuda.lang::do-var-types code nil nil)    '((a :variable int) (b :variable int)))
    (is (cl-cuda.lang::do-binding-var binding)       'a)
    (is (cl-cuda.lang::do-binding-type binding var-env nil) 'int)
    (is (cl-cuda.lang::do-binding-init-form binding) 0)
    (is (cl-cuda.lang::do-binding-step-form binding) '(+ a 1))
    (is (cl-cuda.lang::do-test-form code)            '(> a 15))
    (is (cl-cuda.lang::do-statements code)           '((return)))))

;; test compile-do
(let ((lisp-code '(do ((a 0 (+ a 1))
                       (b 0 (+ b 1)))
                      ((> a 15))
                    (return)))
      (c-code (cl-cuda.lang::unlines "for ( int a = 0, int b = 0; ! (a > 15); a = (a + 1), b = (b + 1) )"
                                "{"
                                "  return;"
                                "}")))
  (cl-cuda.lang::with-variable-environment (var-env ((a :variable int) (b :variable int)))
    (is (cl-cuda.lang::compile-do-init-part lisp-code nil     nil) "int a = 0, int b = 0")
    (is (cl-cuda.lang::compile-do-test-part lisp-code var-env nil) "! (a > 15)")
    (is (cl-cuda.lang::compile-do-step-part lisp-code var-env nil) "a = (a + 1), b = (b + 1)")
    (is (cl-cuda.lang::compile-do lisp-code nil nil) c-code)))

(let ((lisp-code '(do ((a 0.0 (+ a 1.0)))
                      ((> a 15.0))
                    (return)))
      (c-code (cl-cuda.lang::unlines "for ( float a = 0.0; ! (a > 15.0); a = (a + 1.0) )"
                                "{"
                                "  return;"
                                "}")))
  (is (cl-cuda.lang::compile-do lisp-code nil nil) c-code))

(let ((lisp-code '(do ((a 0)) ((> a 10)) (return)))
      (c-code (cl-cuda.lang::unlines "for ( int a = 0; ! (a > 10);  )"
                                "{"
                                "  return;"
                                "}")))
  (is (cl-cuda.lang::compile-do lisp-code nil nil) c-code))


;;;
;;; test compile-with-shared-memory
;;;

(diag "test compile-with-shared-memory")

;; test with-shared-memory-p
(is (cl-cuda.lang::with-shared-memory-p '(with-shared-memory ((a float 16))
                                      (return)))
    t)
(is (cl-cuda.lang::with-shared-memory-p '(with-shared-memory () (return))) t)
(is (cl-cuda.lang::with-shared-memory-p '(with-shared-memory ())         ) t)
(is (cl-cuda.lang::with-shared-memory-p '(with-shared-memory)            ) t)

;; test compile-with-shared-memory
(let ((lisp-code '(with-shared-memory ((a int 16)
                                       (b float 16 16))
                   (return)))
      (c-code (cl-cuda.lang::unlines "{"
                                "  __shared__ int a[16];"
                                "  __shared__ float b[16][16];"
                                "  return;"
                                "}")))
  (is (cl-cuda.lang::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory () (return)))
      (c-code (cl-cuda.lang::unlines "{"
                                "  return;"
                                "}")))
  (is (cl-cuda.lang::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory ()))
      (c-code (cl-cuda.lang::unlines "{"
                                ""
                                "}")))
  (is (cl-cuda.lang::compile-with-shared-memory lisp-code nil nil) c-code))

(is-error (cl-cuda.lang::compile-with-shared-memory '(with-shared-memory) nil nil) simple-error)

(let ((lisp-code '(with-shared-memory ((a float))
                    (return)))
      (c-code (cl-cuda.lang::unlines "{"
                                "  __shared__ float a;"
                                "  return;"
                                "}")))
  (is (cl-cuda.lang::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory (a float)
                    (return))))
  (is-error (cl-cuda.lang::compile-with-shared-memory lisp-code nil nil) simple-error))

(let ((lisp-code '(with-shared-memory ((a float 16 16))
                    (set (aref a 0 0) 1.0)))
      (c-code (cl-cuda.lang::unlines "{"
                                "  __shared__ float a[16][16];"
                                "  a[0][0] = 1.0;"
                                "}")))
  (is (cl-cuda.lang::compile-with-shared-memory lisp-code nil nil) c-code))

(let ((lisp-code '(with-shared-memory ((a float 16 16))
                    (set (aref a 0) 1.0))))
  (is-error (cl-cuda.lang::compile-with-shared-memory lisp-code nil nil) simple-error))


;;;
;;; test compile-set
;;;

(diag "test compile-set")

;; test set-p
(is (cl-cuda.lang::set-p '(set x          1)) t)
(is (cl-cuda.lang::set-p '(set (aref x i) 1)) t)

;; test compile-set
(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)))
  (is (cl-cuda.lang::compile-set '(set x 1) var-env nil) "x = 1;")
  (is-error (cl-cuda.lang::compile-set '(set x 1.0) var-env nil) simple-error))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable int*)))
  (is (cl-cuda.lang::compile-set '(set (aref x 0) 1) var-env nil) "x[0] = 1;")
  (is-error (cl-cuda.lang::compile-set '(set (aref x 0) 1.0) var-env nil) simple-error))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable float3)))
  (is (cl-cuda.lang::compile-set '(set (float3-x x) 1.0) var-env nil) "x.x = 1.0;")
  (is-error (cl-cuda.lang::compile-set '(set (float3-x x) 1) var-env nil) simple-error))


;;;
;;; test compile-place (not implemented)
;;;

 

;;;
;;; test compile-progn (not implemented)
;;;



;;;
;;; test compile-return (not implemented)
;;;



;;;
;;; test compile-syncthreads
;;;

(diag "test compile-syncthreads")

;; test syncthreads-p
(is (cl-cuda.lang::syncthreads-p '(syncthreads)) t)

;; test compile-syncthreads
(is (cl-cuda.lang::compile-syncthreads '(syncthreads)) "__syncthreads();")


;;;
;;; test compile-function
;;;

(diag "test compile-function")

;; test built-in-function-p
(is (cl-cuda.lang::built-in-function-p '(cl-cuda.lang::%add 1 1)) t)
(is (cl-cuda.lang::built-in-function-p '(cl-cuda.lang::%sub 1 1)) t)
(is (cl-cuda.lang::built-in-function-p '(foo 1 1)) nil)

;; test function-candidates
(is       (cl-cuda.lang::function-candidates 'cl-cuda.lang::%add) '(((int int) int t "+")
                                                          ((float float) float t "+")
                                                          ((float3 float3) float3 nil "float3_add")
                                                          ((float4 float4) float4 nil "float4_add")
                                                          ((double double) double t "+")
                                                          ((double3 double3) double3 nil "double3_add")
                                                          ((double4 double4) double4 nil "double4_add")))
(is-error (cl-cuda.lang::function-candidates 'foo) simple-error)

;; test built-in-function-argument-types
(is       (cl-cuda.lang::built-in-function-argument-types '(cl-cuda.lang::%add 1 1) nil nil) '(int int))
(is       (cl-cuda.lang::built-in-function-argument-types '(expt 1.0 1.0) nil nil) '(float float))
(is-error (cl-cuda.lang::built-in-function-argument-types '(foo) nil nil) simple-error)

;; test built-in-function-return-type
(is       (cl-cuda.lang::built-in-function-return-type '(cl-cuda.lang::%add 1 1) nil nil) 'int)
(is       (cl-cuda.lang::built-in-function-return-type '(expt 1.0 1.0) nil nil) 'float)
(is-error (cl-cuda.lang::built-in-function-return-type '(foo) nil nil) simple-error)

;; test built-in-function-infix-p
(is       (cl-cuda.lang::built-in-function-infix-p '(cl-cuda.lang::%add 1 1) nil nil) t)
(is       (cl-cuda.lang::built-in-function-infix-p '(expt 1.0 1.0) nil nil) nil)
(is-error (cl-cuda.lang::built-in-function-infix-p '(foo) nil nil) simple-error)

;; test built-in-function-prefix-p
(is       (cl-cuda.lang::built-in-function-prefix-p '(cl-cuda.lang::%add 1 1) nil nil) nil)
(is       (cl-cuda.lang::built-in-function-prefix-p '(expt 1.0 1.0) nil nil) t)
(is-error (cl-cuda.lang::built-in-function-prefix-p '(foo) nil nil) simple-error)

;; test built-in-function-c-string
(is       (cl-cuda.lang::built-in-function-c-string '(cl-cuda.lang::%add 1 1) nil nil) "+")
(is       (cl-cuda.lang::built-in-function-c-string '(expt 1.0 1.0) nil nil) "powf")
(is-error (cl-cuda.lang::built-in-function-c-string '(foo) nil nil) simple-error)

;; test function-p
(is (cl-cuda.lang::function-p 'a        ) nil)
(is (cl-cuda.lang::function-p '()       ) nil)
(is (cl-cuda.lang::function-p '1        ) nil)
(is (cl-cuda.lang::function-p '(foo)    ) t  )
(is (cl-cuda.lang::function-p '(+ 1 1)  ) t  )
(is (cl-cuda.lang::function-p '(foo 1 1)) t  )

;; test function-operator
(is-error (cl-cuda.lang::function-operator 'a        ) simple-error)
(is       (cl-cuda.lang::function-operator '(foo)    ) 'foo        )
(is       (cl-cuda.lang::function-operator '(+ 1 1)  ) '+          )
(is       (cl-cuda.lang::function-operator '(foo 1 1)) 'foo        )

;; test function-operands
(is-error (cl-cuda.lang::function-operands 'a        ) simple-error)
(is       (cl-cuda.lang::function-operands '(foo)    ) '()         )
(is       (cl-cuda.lang::function-operands '(+ 1 1)  ) '(1 1)      )
(is       (cl-cuda.lang::function-operands '(foo 1 1)) '(1 1)      )

;; test compile-function
(is-error (cl-cuda.lang::compile-function 'a                   nil nil) simple-error)
(is       (cl-cuda.lang::compile-function '(cl-cuda.lang::%add 1 1) nil nil) "(1 + 1)"   )

(is (cl-cuda.lang::compile-function '(cl-cuda.lang::%negate 1)   nil nil) "int_negate (1)")
(is (cl-cuda.lang::compile-function '(cl-cuda.lang::%negate 1.0) nil nil) "float_negate (1.0)")
(is (cl-cuda.lang::compile-function '(cl-cuda.lang::%negate (float3 1.0 1.0 1.0)) nil nil)
    "float3_negate (make_float3 (1.0, 1.0, 1.0))")

(is (cl-cuda.lang::compile-function '(cl-cuda.lang::%recip 2)   nil nil) "int_recip (2)")
(is (cl-cuda.lang::compile-function '(cl-cuda.lang::%recip 2.0) nil nil) "float_recip (2.0)")
(is (cl-cuda.lang::compile-function '(cl-cuda.lang::%recip (float3 2.0 2.0 2.0)) nil nil)
    "float3_recip (make_float3 (2.0, 2.0, 2.0))")

(cl-cuda.lang::with-function-environment (func-env ((foo :function void () ())))
  (is (cl-cuda.lang::compile-function '(foo) nil func-env :statement-p t) "cl_cuda_test_lang_foo ();"))

(cl-cuda.lang::with-function-environment (func-env ((foo :function void ((x int) (y int)) ())))
  (is-error (cl-cuda.lang::compile-function '(foo 1 1)   nil nil) simple-error)
  (is       (cl-cuda.lang::compile-function '(foo 1 1)   nil func-env :statement-p t) "cl_cuda_test_lang_foo (1, 1);")
  (is-error (cl-cuda.lang::compile-function '(foo 1 1 1) nil func-env :statement-p t) simple-error))

(is (cl-cuda.lang::compile-function '(float3 1.0 1.0 1.0)     nil nil) "make_float3 (1.0, 1.0, 1.0)"     )
(is (cl-cuda.lang::compile-function '(float4 1.0 1.0 1.0 1.0) nil nil) "make_float4 (1.0, 1.0, 1.0, 1.0)")

(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)))
  (is (cl-cuda.lang::compile-function '(pointer x) var-env nil) "& (x)"))

(is (cl-cuda.lang::compile-function '(floor 1.0) nil nil) "floorf (1.0)")

(is (cl-cuda.lang::compile-function '(sqrt 1.0) nil nil) "sqrtf (1.0)")


;;;
;;; test compile-macro
;;;

(diag "test compile-macro")

;; test macro-form-p
(cl-cuda.lang::with-function-environment (func-env ((foo :macro (x) (`(progn ,x)))))
  (is (cl-cuda.lang::macro-form-p '(+ 1 1) func-env) t)
  (is (cl-cuda.lang::macro-form-p '(foo 1) func-env) t)
  (is (cl-cuda.lang::macro-form-p 'bar func-env) nil))

;; test macro-operator
(is (cl-cuda.lang::macro-operator '(+ 1 1) (cl-cuda.lang::empty-function-environment)) '+)

;; test macro-operands
(is (cl-cuda.lang::macro-operands '(+ 1 1) (cl-cuda.lang::empty-function-environment)) '(1 1))


;;;
;;; test compile-literal
;;;

(diag "test compile-literal")

;; test LITERAL-P function
(is (cl-cuda.lang::literal-p 't)    t)
(is (cl-cuda.lang::literal-p 'nil)  t)
(is (cl-cuda.lang::literal-p 1)     t)
(is (cl-cuda.lang::literal-p 1.0)   t)
(is (cl-cuda.lang::literal-p 1.0d0) t)

;; test COMPILE-LITERAL function
(is       (cl-cuda.lang::compile-literal 't)    "true")
(is       (cl-cuda.lang::compile-literal 'nil)  "false")
(is       (cl-cuda.lang::compile-literal 1)     "1")
(is       (cl-cuda.lang::compile-literal 1.0)   "1.0")
(is       (cl-cuda.lang::compile-literal 1.0d0) "(double)1.0")


;;;
;;; test compile-symbol
;;;

(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)
                                              (y :constant float)
                                              (z :symbol-macro expanded-z)
                                              (expanded-z :constant float3)))
  (is (cl-cuda.lang::symbol-p 'x) t)
  (is (cl-cuda.lang::symbol-p 'y) t)
  (is (cl-cuda.lang::symbol-p 'z) t)
  (is (cl-cuda.lang::symbol-p 'a) t)
  (is (cl-cuda.lang::compile-symbol 'x var-env nil) "x")
  (is (cl-cuda.lang::compile-symbol 'y var-env nil) "y")
  (is (cl-cuda.lang::compile-symbol 'z var-env nil) "expanded_z")
  (is-error (cl-cuda.lang::compile-symbol 'a var-env nil) simple-error))


;;;
;;; test compile-cuda-dimension (not implemented)
;;;



;;;
;;; test compile-variable-reference
;;;

(diag "test compile-variable-reference")

;; test variable-reference-p
(is (cl-cuda.lang::variable-reference-p '(aref x))       t)
(is (cl-cuda.lang::variable-reference-p '(aref x i))     t)
(is (cl-cuda.lang::variable-reference-p '(aref x i i))   t)
(is (cl-cuda.lang::variable-reference-p '(aref x i i i)) t)
(is (cl-cuda.lang::variable-reference-p '(float3-x x))   t)
(is (cl-cuda.lang::variable-reference-p '(float3-y x))   t)
(is (cl-cuda.lang::variable-reference-p '(float3-z x))   t)
(is (cl-cuda.lang::variable-reference-p '(float4-x x))   t)
(is (cl-cuda.lang::variable-reference-p '(float4-y x))   t)
(is (cl-cuda.lang::variable-reference-p '(float4-z x))   t)
(is (cl-cuda.lang::variable-reference-p '(float4-w x))   t)

;; test compile-variable-reference
(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)))
  (is-error (cl-cuda.lang::compile-variable-reference '(aref x)   var-env nil) simple-error)
  (is-error (cl-cuda.lang::compile-variable-reference '(aref x 0) var-env nil) simple-error))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable int*)))
  (is       (cl-cuda.lang::compile-variable-reference '(aref x 0)   var-env nil) "x[0]"      )
  (is-error (cl-cuda.lang::compile-variable-reference '(aref x 0 0) var-env nil) simple-error))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable int**)))
  (is-error (cl-cuda.lang::compile-variable-reference '(aref x 0)   var-env nil) simple-error)
  (is       (cl-cuda.lang::compile-variable-reference '(aref x 0 0) var-env nil) "x[0][0]"   ))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable float3)))
  (is (cl-cuda.lang::compile-variable-reference '(float3-x x) var-env nil) "x.x")
  (is (cl-cuda.lang::compile-variable-reference '(float3-y x) var-env nil) "x.y")
  (is (cl-cuda.lang::compile-variable-reference '(float3-z x) var-env nil) "x.z"))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable float4)))
  (is (cl-cuda.lang::compile-variable-reference '(float4-x x) var-env nil) "x.x")
  (is (cl-cuda.lang::compile-variable-reference '(float4-y x) var-env nil) "x.y")
  (is (cl-cuda.lang::compile-variable-reference '(float4-z x) var-env nil) "x.z")
  (is (cl-cuda.lang::compile-variable-reference '(float4-w x) var-env nil) "x.w"))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable float)))
  (is-error (cl-cuda.lang::compile-variable-reference '(float3-x x) var-env nil) simple-error))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable float3*)))
  (is (cl-cuda.lang::compile-vector-variable-reference '(float3-x (aref x 0)) var-env nil) "x[0].x"))


;;;
;;; test compile-inline-if
;;;

(diag "test compile-inline-if")

;; test inline-if-p
(is (cl-cuda.lang::inline-if-p '(if)) nil)
(is (cl-cuda.lang::inline-if-p '(if t)) nil)
(is (cl-cuda.lang::inline-if-p '(if t 2)) nil)
(is (cl-cuda.lang::inline-if-p '(if t 2 3)) t)
(is (cl-cuda.lang::inline-if-p '(if t 2 3 4)) nil)

;; test compile-inline-if
(is-error (cl-cuda.lang::compile-inline-if '(if) nil nil) simple-error)
(is-error (cl-cuda.lang::compile-inline-if '(if (= 1 1)) nil nil) simple-error)
(is-error (cl-cuda.lang::compile-inline-if '(if (= 1 1) 1) nil nil) simple-error)
(is       (cl-cuda.lang::compile-inline-if '(if (= 1 1) 1 2) nil nil) "((1 == 1) ? 1 : 2)")
(is-error (cl-cuda.lang::compile-inline-if '(if (= 1 1) 1 2 3) nil nil) simple-error)
(is-error (cl-cuda.lang::compile-inline-if '(if 1 2 3) nil nil) simple-error)
(is-error (cl-cuda.lang::compile-inline-if '(if t 1 1.0) nil nil) simple-error)


;;;
;;; test type-of-expression
;;;

(diag "test type-of-expression")

;; test type-of-expression
(is (cl-cuda.lang::type-of-expression 't   nil nil) 'bool )
(is (cl-cuda.lang::type-of-expression 'nil nil nil) 'bool )
(is (cl-cuda.lang::type-of-expression '1   nil nil) 'int  )
(is (cl-cuda.lang::type-of-expression '1.0 nil nil) 'float)

;; test type-of-literal
(is       (cl-cuda.lang::type-of-literal 't    ) 'bool       )
(is       (cl-cuda.lang::type-of-literal 'nil  ) 'bool       )
(is       (cl-cuda.lang::type-of-literal '1    ) 'int        )
(is       (cl-cuda.lang::type-of-literal '1.0  ) 'float      )
(is       (cl-cuda.lang::type-of-literal '1.0d0) 'double     )

;; test type-of-symbol
(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)
                                              (y :constant float)
                                              (z :symbol-macro expanded-z)
                                              (expanded-z :constant float3)))
  (is (cl-cuda.lang::type-of-symbol 'x var-env nil) 'int)
  (is (cl-cuda.lang::type-of-symbol 'y var-env nil) 'float)
  (is (cl-cuda.lang::type-of-symbol 'z var-env nil) 'float3)
  (is-error (cl-cuda.lang::type-of-symbol 'a var-env nil) simple-error))

;; test type-of-function
(cl-cuda.lang::with-function-environment (func-env ((foo :function int ((x int) (y int)) ())))
  (is (cl-cuda.lang::type-of-function '(cl-cuda.lang::%add 1 1) nil func-env) 'int)
  (is (cl-cuda.lang::type-of-function '(foo 1 1) nil func-env) 'int))

;; test type-of-function
(is       (cl-cuda.lang::type-of-function '(cl-cuda.lang::%add 1 1)     nil nil) 'int        )
(is       (cl-cuda.lang::type-of-function '(cl-cuda.lang::%add 1.0 1.0) nil nil) 'float      )
(is-error (cl-cuda.lang::type-of-function '(cl-cuda.lang::%add 1 1.0)   nil nil) simple-error)
(is       (cl-cuda.lang::type-of-function '(expt 1.0 1.0)          nil nil) 'float      )

;; test type-of-expression for grid, block and thread
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::grid-dim-x   nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::grid-dim-y   nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::grid-dim-z   nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::block-idx-x  nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::block-idx-y  nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::block-idx-z  nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::block-dim-x  nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::block-dim-y  nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::block-dim-z  nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::thread-idx-x nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::thread-idx-y nil nil) 'int)
(is (cl-cuda.lang::type-of-expression 'cl-cuda.lang::thread-idx-z nil nil) 'int)

;; test type-of-variable-reference
(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)))
  (is-error (cl-cuda.lang::type-of-variable-reference '(aref x) var-env nil) simple-error))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable int*)))
  (is       (cl-cuda.lang::type-of-variable-reference '(aref x 0)   var-env nil) 'int        )
  (is-error (cl-cuda.lang::type-of-variable-reference '(aref x 0 0) var-env nil) simple-error))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable int**)))
  (is-error (cl-cuda.lang::type-of-variable-reference '(aref x 0)   var-env nil) simple-error)
  (is       (cl-cuda.lang::type-of-variable-reference '(aref x 0 0) var-env nil) 'int        ))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable float3)))
  (is (cl-cuda.lang::type-of-variable-reference '(float3-x x) var-env nil) 'float)
  (is (cl-cuda.lang::type-of-variable-reference '(float3-y x) var-env nil) 'float)
  (is (cl-cuda.lang::type-of-variable-reference '(float3-z x) var-env nil) 'float))

(cl-cuda.lang::with-variable-environment (var-env ((x :variable float4)))
  (is (cl-cuda.lang::type-of-variable-reference '(float4-x x) var-env nil) 'float)
  (is (cl-cuda.lang::type-of-variable-reference '(float4-y x) var-env nil) 'float)
  (is (cl-cuda.lang::type-of-variable-reference '(float4-z x) var-env nil) 'float)
  (is (cl-cuda.lang::type-of-variable-reference '(float4-w x) var-env nil) 'float))

;; test type-of-inline-if
(is-error (cl-cuda.lang::type-of-inline-if '(if) nil nil) simple-error)
(is-error (cl-cuda.lang::type-of-inline-if '(if (= 1 1)) nil nil) simple-error)
(is-error (cl-cuda.lang::type-of-inline-if '(if (= 1 1) 1) nil nil) simple-error)
(is       (cl-cuda.lang::type-of-inline-if '(if (= 1 1) 1 2) nil nil) 'int)
(is-error (cl-cuda.lang::type-of-inline-if '(if (= 1 1) 1 2 3) nil nil) simple-error)
(is-error (cl-cuda.lang::type-of-inline-if '(if 1 2 3) nil nil) simple-error)
(is-error (cl-cuda.lang::type-of-inline-if '(if (= 1 1) 1 2.0) nil nil) simple-error)


;;;
;;; test variable environment
;;;

(diag "test variable environment")

;; test variable environment elements
(let ((var     (cl-cuda.lang::make-varenv-variable 'x 'int))
      (const   (cl-cuda.lang::make-varenv-constant 'y 'float))
      (sym-mac (cl-cuda.lang::make-varenv-symbol-macro 'z '(expanded z))))
  ;; test varenv-name
  (is       (cl-cuda.lang::varenv-name var) 'x)
  (is       (cl-cuda.lang::varenv-name const) 'y)
  (is       (cl-cuda.lang::varenv-name sym-mac) 'z)
  (is-error (cl-cuda.lang::varenv-name nil) simple-error)
  ;; test variable
  (is       (cl-cuda.lang::varenv-variable-p var) t)
  (is       (cl-cuda.lang::varenv-variable-name var) 'x)
  (is       (cl-cuda.lang::varenv-variable-type var) 'int)
  (is       (cl-cuda.lang::varenv-variable-p const) nil)
  (is-error (cl-cuda.lang::varenv-variable-name const) simple-error)
  (is-error (cl-cuda.lang::varenv-variable-type const) simple-error)
  ;; test constant
  (is       (cl-cuda.lang::varenv-constant-p const) t)
  (is       (cl-cuda.lang::varenv-constant-name const) 'y)
  (is       (cl-cuda.lang::varenv-constant-type const) 'float)
  (is       (cl-cuda.lang::varenv-constant-p var) nil)
  (is-error (cl-cuda.lang::varenv-constant-name var) simple-error)
  (is-error (cl-cuda.lang::varenv-constant-type var) simple-error)
  ;; test symbol macro
  (is       (cl-cuda.lang::varenv-symbol-macro-p sym-mac) t)
  (is       (cl-cuda.lang::varenv-symbol-macro-name sym-mac) 'z)
  (is       (cl-cuda.lang::varenv-symbol-macro-expansion sym-mac) '(expanded z))
  (is       (cl-cuda.lang::varenv-symbol-macro-p var) nil)
  (is-error (cl-cuda.lang::varenv-symbol-macro-name var) simple-error)
  (is-error (cl-cuda.lang::varenv-symbol-macro-expansion var) simple-error))

;; test making varialbe environment elements
(is-error (cl-cuda.lang::make-varenv-variable '(x) 'int) simple-error)
(is-error (cl-cuda.lang::make-varenv-variable '(x) 'invalid-type) simple-error)
(is-error (cl-cuda.lang::make-varenv-constant '(x) 'int) simple-error)
(is-error (cl-cuda.lang::make-varenv-constant '(x) 'invalid-type) simple-error)
(is-error (cl-cuda.lang::make-varenv-symbol-macro '(x) 'int) simple-error)
(is-error (cl-cuda.lang::bulk-add-variable-environment '((x :invalid-keyword int)) (cl-cuda.lang::empty-variable-environment))
          simple-error)

;; test variable environment
(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)
                                              (y :constant float)
                                              (z :symbol-macro (expanded-z))))
  ;; test predicates
  (is       (cl-cuda.lang::variable-environment-variable-exists-p 'x var-env) t)
  (is       (cl-cuda.lang::variable-environment-variable-exists-p 'y var-env) nil)
  (is       (cl-cuda.lang::variable-environment-constant-exists-p 'y var-env) t)
  (is       (cl-cuda.lang::variable-environment-constant-exists-p 'x var-env) nil)
  (is       (cl-cuda.lang::variable-environment-symbol-macro-exists-p 'z var-env) t)
  (is       (cl-cuda.lang::variable-environment-symbol-macro-exists-p 'x var-env) nil)
  ;; test selectors
  (is       (cl-cuda.lang::variable-environment-type-of-variable 'x var-env) 'int)
  (is-error (cl-cuda.lang::variable-environment-type-of-variable 'y var-env) simple-error)
  (is       (cl-cuda.lang::variable-environment-type-of-constant 'y var-env) 'float)
  (is-error (cl-cuda.lang::variable-environment-type-of-constant 'x var-env) simple-error)
  (is       (cl-cuda.lang::variable-environment-symbol-macro-expansion 'z var-env) '(expanded-z))
  (is-error (cl-cuda.lang::variable-environment-symbol-macro-expansion 'x var-env) simple-error))

;; test shadowing in variable environment
(cl-cuda.lang::with-variable-environment (var-env ((x :variable int)
                                              (x :symbol-macro (expanded-x))))
  (is       (cl-cuda.lang::variable-environment-variable-exists-p 'x var-env) nil)
  (is-error (cl-cuda.lang::variable-environment-type-of-variable 'x var-env) simple-error)
  (is       (cl-cuda.lang::variable-environment-symbol-macro-exists-p 'x var-env) t)
  (is       (cl-cuda.lang::variable-environment-symbol-macro-expansion 'x var-env) '(expanded-x)))

;; test making variable environment with kernel definition
(cl-cuda.lang::with-kernel-definition (def ((f :function void () ((return)))
                                       (x :symbol-macro 1.0)
                                       (y :constant float 1.0)))
  (let ((var-env (cl-cuda.lang::make-variable-environment-with-kernel-definition 'f def)))
    (is (cl-cuda.lang::variable-environment-constant-exists-p 'y var-env) t)
    (is (cl-cuda.lang::variable-environment-type-of-constant 'y var-env) 'float)
    (is (cl-cuda.lang::variable-environment-symbol-macro-exists-p 'x var-env) t)
    (is (cl-cuda.lang::variable-environment-symbol-macro-expansion 'x var-env) 1.0)))


;;;
;;; test function environment
;;;

(diag "test function environment")

;; test function environment elements
(let ((func  (cl-cuda.lang::make-funcenv-function 'f 'int '((x int)) '((return x))))
      (macro (cl-cuda.lang::make-funcenv-macro 'g '(x) '(`(expanded ,x)) (lambda (args)
                                                                      (destructuring-bind (x) args
                                                                        `(expanded ,x))))))
  ;; test funcenv-name
  (is       (cl-cuda.lang::funcenv-name func) 'f)
  (is       (cl-cuda.lang::funcenv-name macro) 'g)
  (is-error (cl-cuda.lang::funcenv-name nil) simple-error)
  ;; test function
  (is       (cl-cuda.lang::funcenv-function-p func) t)
  (is       (cl-cuda.lang::funcenv-function-name func) 'f)
  (is       (cl-cuda.lang::funcenv-function-return-type func) 'int)
  (is       (cl-cuda.lang::funcenv-function-arguments func) '((x int)))
  (is       (cl-cuda.lang::funcenv-function-body func) '((return x)))
  (is       (cl-cuda.lang::funcenv-function-p macro) nil)
  (is-error (cl-cuda.lang::funcenv-function-name macro) simple-error)
  (is-error (cl-cuda.lang::funcenv-function-return-type macro) simple-error)
  (is-error (cl-cuda.lang::funcenv-function-arguments macro) simple-error)
  (is-error (cl-cuda.lang::funcenv-function-body macro) simple-error)
  ;; test macro
  (is       (cl-cuda.lang::funcenv-macro-p macro) t)
  (is       (cl-cuda.lang::funcenv-macro-name macro) 'g)
  (is       (cl-cuda.lang::funcenv-macro-arguments macro) '(x))
  (is       (cl-cuda.lang::funcenv-macro-body macro) '(`(expanded ,x)))
  (is       (funcall (cl-cuda.lang::funcenv-macro-expander macro) '(x)) '(expanded x))
  (is       (cl-cuda.lang::funcenv-macro-p func) nil)
  (is-error (cl-cuda.lang::funcenv-macro-name func) simple-error)
  (is-error (cl-cuda.lang::funcenv-macro-arguments func) simple-error)
  (is-error (cl-cuda.lang::funcenv-macro-body func) simple-error)
  (is-error (cl-cuda.lang::funcenv-macro-expander func) simple-error))

;; test making function environment elements
(is-error (cl-cuda.lang::make-funcenv-function '(f) 'int '((x int)) '((return x))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-function '(g) '(x) '`(expanded ,x) (lambda (args) (destructuring-bind (x) args `(expanded ,x)))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-function 'f 'invalid-type '((x int)) '((return x))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-function 'f 'int '(x) '((return x))) type-error)
(is-error (cl-cuda.lang::make-funcenv-function 'f 'int '(((x) int)) '((return x))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-function 'f 'int '((x invalid-type)) '((return x))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-function 'f 'int '((x int y)) '((return x))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-function 'f 'int '((x int)) 'x) simple-error)
(is-error (cl-cuda.lang::make-funcenv-macro 'g 'x '(`(expanded ,x)) (lambda (args) (destructuring-bind (x) args `(expanded ,x)))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-macro 'g '(x) 'x (lambda (args) (destructuring-bind (x) args x))) simple-error)
(is-error (cl-cuda.lang::make-funcenv-macro 'g '(x) '(`(expanded ,x)) nil) simple-error)
(is-error (cl-cuda.lang::bulk-add-function-environment '((f :invalid-keyword int ((x int)) ((return x))))
                                                  (cl-cuda.lang::empty-function-environment))
          simple-error)

;; test function environment
(cl-cuda.lang::with-function-environment (func-env ((f :function int ((x int)) ((return x)))
                                               (g :macro (x) (`(expanded ,x)))))
  ;; test predicates
  (is       (cl-cuda.lang::function-environment-function-exists-p 'f func-env) t)
  (is       (cl-cuda.lang::function-environment-function-exists-p 'g func-env) nil)
  (is       (cl-cuda.lang::function-environment-macro-exists-p 'g func-env) t)
  (is       (cl-cuda.lang::function-environment-macro-exists-p 'f func-env) nil)
  ;; test selectors
  (is       (cl-cuda.lang::function-environment-function-c-name 'f func-env) "cl_cuda_test_lang_f")
  (is-error (cl-cuda.lang::function-environment-function-c-name 'g func-env) simple-error)
  (is       (cl-cuda.lang::function-environment-function-return-type 'f func-env) 'int)
  (is-error (cl-cuda.lang::function-environment-function-return-type 'g func-env) simple-error)
  (is       (cl-cuda.lang::function-environment-function-arguments 'f func-env) '((x int)))
  (is-error (cl-cuda.lang::function-environment-function-arguments 'g func-env) simple-error)
  (is       (cl-cuda.lang::function-environment-function-argument-types 'f func-env) '(int))
  (is-error (cl-cuda.lang::function-environment-function-argument-types 'g func-env) simple-error)
  (is       (cl-cuda.lang::function-environment-function-body 'f func-env) '((return x)))
  (is-error (cl-cuda.lang::function-environment-function-body 'g func-env) simple-error)
  (is       (cl-cuda.lang::function-environment-macro-arguments 'g func-env) '(x))
  (is-error (cl-cuda.lang::function-environment-macro-arguments 'f func-env) simple-error)
  (is       (cl-cuda.lang::function-environment-macro-body 'g func-env) '(`(expanded ,x)))
  (is-error (cl-cuda.lang::function-environment-macro-body 'f func-env) simple-error)
  (is       (funcall (cl-cuda.lang::function-environment-macro-expander 'g func-env) '(x)) '(expanded x))
  (is-error (cl-cuda.lang::function-environment-macro-expander 'f func-env) simple-error))

;; test shadowing in function environment
(cl-cuda.lang::with-function-environment (func-env ((f :function int ((x int)) ((return x)))
                                               (f :macro (x) (`(expanded ,x)))))
  (is       (cl-cuda.lang::function-environment-function-exists-p 'f func-env) nil)
  (is-error (cl-cuda.lang::function-environment-function-return-type 'f func-env) simple-error)
  (is-error (cl-cuda.lang::function-environment-function-arguments 'f func-env) simple-error)
  (is-error (cl-cuda.lang::function-environment-function-body 'f func-env) simple-error)
  (is       (cl-cuda.lang::function-environment-macro-exists-p 'f func-env) t)
  (is       (cl-cuda.lang::function-environment-macro-arguments 'f func-env) '(x))
  (is       (cl-cuda.lang::function-environment-macro-body 'f func-env)  '(`(expanded ,x)))
  (is       (funcall (cl-cuda.lang::function-environment-macro-expander 'f func-env) '(x)) '(expanded x)))

;; test making function environment with kernel definition
(cl-cuda.lang::with-kernel-definition (def ((f :function int ((x int)) ((return x)))
                                       (g :macro (x) (`(expanded ,x)))))
  (let ((func-env (cl-cuda.lang::make-function-environment-with-kernel-definition def)))
    (is (cl-cuda.lang::function-environment-function-exists-p 'f func-env) t)
    (is (cl-cuda.lang::function-environment-function-return-type 'f func-env) 'int)
    (is (cl-cuda.lang::function-environment-function-arguments 'f func-env) '((x int)))
    (is (cl-cuda.lang::function-environment-function-argument-types 'f func-env) '(int))
    (is (cl-cuda.lang::function-environment-function-body 'f func-env) '((return x)))
    (is (cl-cuda.lang::function-environment-macro-exists-p 'g func-env) t)
    (is (cl-cuda.lang::function-environment-macro-arguments 'g func-env) '(x))
    (is (cl-cuda.lang::function-environment-macro-body 'g func-env) '(`(expanded ,x)))
    (is (funcall (cl-cuda.lang::function-environment-macro-expander 'g func-env) '(x)) '(expanded x))))


;;;
;;; test utilities
;;;

;; test cl-cuda-symbolicate
(is (cl-cuda.lang::cl-cuda-symbolicate 'a   ) 'cl-cuda.lang::a )
(is (cl-cuda.lang::cl-cuda-symbolicate 'a 'b) 'cl-cuda.lang::ab)


(finalize)
