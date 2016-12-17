#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.compiler.compile-kernel
  (:use :cl
        :cl-cuda.lang.util
        :cl-cuda.lang.type
        :cl-cuda.lang.syntax
        :cl-cuda.lang.environment
        :cl-cuda.lang.kernel
        :cl-cuda.lang.compiler.compile-data
        :cl-cuda.lang.compiler.compile-type
        :cl-cuda.lang.compiler.compile-expression
        :cl-cuda.lang.compiler.compile-statement
        :cl-cuda.lang.compiler.type-of-expression)
  (:export :compile-kernel))
(in-package :cl-cuda.lang.compiler.compile-kernel)


;;;
;;; Kernel to Environment
;;;

(defun %add-function-arguments (kernel name var-env)
  (flet ((aux (var-env0 argument)
           (let ((var (argument-var argument))
                 (type (argument-type argument)))
             (variable-environment-add-variable var type var-env0))))
    (reduce #'aux (kernel-function-arguments kernel name)
            :initial-value var-env)))

(defun %add-symbol-macros (kernel var-env)
  (flet ((aux (var-env0 name)
           (let ((expansion (kernel-symbol-macro-expansion kernel name)))
             (variable-environment-add-symbol-macro name expansion
                                                    var-env0))))
    (reduce #'aux (kernel-symbol-macro-names kernel)
            :initial-value var-env)))

(defun %add-globals (kernel var-env)
  (flet ((aux (var-env0 name)
           (let* ((initializer (kernel-global-initializer kernel name))
                  (type (type-of-expression initializer nil nil)))
            (variable-environment-add-global name type initializer var-env0))))
    (reduce #'aux (kernel-global-names kernel)
            :initial-value var-env)))

(defun kernel->variable-environment (kernel name)
  (if name
      (%add-function-arguments kernel name
       (%add-symbol-macros kernel
        (%add-globals kernel
         (empty-variable-environment))))
      (%add-symbol-macros kernel
       (%add-globals kernel
        (empty-variable-environment)))))

(defun %add-functions (kernel func-env)
  (flet ((aux (func-env0 name)
           (let ((return-type (kernel-function-return-type kernel name))
                 (argument-types (kernel-function-argument-types kernel
                                                                 name)))
             (function-environment-add-function name return-type
                                                argument-types func-env0))))
    (reduce #'aux (kernel-function-names kernel)
            :initial-value func-env)))

(defun %add-macros (kernel func-env)
  (flet ((aux (func-env0 name)
           (let ((arguments (kernel-macro-arguments kernel name))
                 (body (kernel-macro-body kernel name)))
             (function-environment-add-macro name arguments body func-env0))))
    (reduce #'aux (kernel-macro-names kernel)
            :initial-value func-env)))

(defun kernel->function-environment (kernel)
  (%add-functions kernel
    (%add-macros kernel
      (empty-function-environment))))


;;;
;;; Compile kernel
;;;

(defun compile-includes ()
  "#include \"int.h\"
#include \"float.h\"
#include \"float3.h\"
#include \"float4.h\"
#include \"double.h\"
#include \"double3.h\"
#include \"double4.h\"
#include \"curand.h\"
")

(defun compile-variable-qualifier (qualifier)
  (format nil "__~A__" (string-downcase (princ-to-string qualifier))))

(defun compile-global (kernel name)
  (let ((c-name (kernel-global-c-name kernel name))
        (qualifiers (kernel-global-qualifiers kernel name))
        (initializer (kernel-global-initializer kernel name)))
    (let ((type1 (compile-type
                  (type-of-expression initializer nil nil)))
          (qualifiers1 (mapcar #'compile-variable-qualifier qualifiers))
          (initializer1 (compile-expression initializer
                         (kernel->variable-environment kernel nil)
                         (kernel->function-environment kernel)
                         t)))
      (format nil "~{~A~^ ~} static ~A ~A~@[ = ~A~];~%"
              qualifiers1 type1 c-name initializer1))))

(defun compile-globals (kernel)
  (flet ((aux (name)
           (compile-global kernel name)))
    (let ((globals (mapcar #'aux (kernel-global-names kernel))))
      (format nil "/**
 *  Kernel globals
 */

~{~A~}" globals))))

(defun compile-specifier (return-type)
  (unless (cl-cuda-type-p return-type)
    (error 'type-error :datum return-type :expected 'cl-cuda-type))
  (if (eq return-type 'void)
      "__global__"
      "__device__"))

(defun compile-argument (argument)
  (let ((var (argument-var argument))
        (type (argument-type argument)))
    (let ((var1 (compile-symbol var))
          (type1 (compile-type type)))
      (format nil "~A ~A" type1 var1))))

(defun compile-arguments (arguments)
  (let ((arguments1 (mapcar #'compile-argument arguments)))
    (if arguments1
        (format nil " ~{~A~^, ~} " arguments1)
        "")))

(defun compile-declaration (kernel name)
  (let ((c-name (kernel-function-c-name kernel name))
        (return-type (kernel-function-return-type kernel name))
        (arguments (kernel-function-arguments kernel name)))
    (let ((specifier (compile-specifier return-type))
          (return-type1 (compile-type return-type))
          (arguments1 (compile-arguments arguments)))
      (format nil "~A ~A ~A(~A)" specifier return-type1 c-name arguments1))))

(defun compile-prototype (kernel name)
  (let ((declaration (compile-declaration kernel name)))
    (format nil "extern \"C\" ~A;~%" declaration)))

(defun compile-prototypes (kernel)
  (flet ((aux (name)
           (compile-prototype kernel name)))
    (let ((prototypes (mapcar #'aux (kernel-function-names kernel))))
      (format nil "/**
 *  Kernel function prototypes
 */

~{~A~}" prototypes))))

(defun compile-statements (kernel name)
  (let ((var-env (kernel->variable-environment kernel name))
        (func-env (kernel->function-environment kernel)))
    (flet ((aux (statement)
             (compile-statement statement var-env func-env)))
      (let ((statements (kernel-function-body kernel name)))
        (format nil "~{~A~}" (mapcar #'aux statements))))))

(defun compile-definition (kernel name)
  (let ((declaration (compile-declaration kernel name))
        (statements (compile-statements kernel name)))
    (let ((statements1 (indent 2 statements)))
      (format nil "~A~%{~%~A}~%" declaration statements1))))

(defun compile-definitions (kernel)
  (flet ((aux (name)
           (compile-definition kernel name)))
    (let ((definitions (mapcar #'aux (kernel-function-names kernel))))
      (format nil "/**
 *  Kernel function definitions
 */

~{~A~^~%~}" definitions))))

(defun compile-kernel (kernel)
  (let ((includes (compile-includes))
        (globals (compile-globals kernel))
        (prototypes (compile-prototypes kernel))
        (definitions (compile-definitions kernel)))
    (format nil "~A~%~%~A~%~%~A~%~%~A" includes
                                       globals
                                       prototypes
                                       definitions)))
