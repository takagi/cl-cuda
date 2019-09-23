#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.compiler.compile-expression
  (:use :cl
        :cl-cuda.lang.type
        :cl-cuda.lang.syntax
        :cl-cuda.lang.environment
        :cl-cuda.lang.built-in
        :cl-cuda.lang.compiler.compile-data
        :cl-cuda.lang.compiler.type-of-expression)
  (:export :compile-expression))
(in-package :cl-cuda.lang.compiler.compile-expression)


;;;
;;; Compile expression
;;;

(defun compile-expression (form var-env func-env &optional initializer-p)
  (cond
    ((%macro-p form func-env)
     (compile-macro form var-env func-env initializer-p))
    ((%symbol-macro-p form var-env)
     (compile-symbol-macro form var-env func-env initializer-p))
    ((literal-p form) (compile-literal form))
    ((cuda-dimension-p form) (compile-cuda-dimension form))
    ((reference-p form) (compile-reference form var-env func-env))
    ((inline-if-p form) (compile-inline-if form var-env func-env))
    ((constructor-p form)
     (compile-constructor form var-env func-env initializer-p))
    ((arithmetic-p form) (compile-arithmetic form var-env func-env))
    ((function-p form) (compile-function form var-env func-env))
    (t (error "The value ~S is an invalid expression." form))))


;;;
;;; Macro
;;;

(defun %macro-p (form func-env)
  (cl-cuda.lang.compiler.type-of-expression::%macro-p form func-env))

(defun compile-macro (form var-env func-env initializer-p)
  (let ((operator (macro-operator form))
        (operands (macro-operands form)))
    (let ((expander (function-environment-macro-expander func-env operator)))
      (let ((form1 (funcall expander operands)))
        (compile-expression form1 var-env func-env initializer-p)))))


;;;
;;; Symbol macro
;;;

(defun %symbol-macro-p (form var-env)
  (cl-cuda.lang.compiler.type-of-expression::%symbol-macro-p form var-env))

(defun compile-symbol-macro (form var-env func-env initializer-p)
  (let ((form1 (variable-environment-symbol-macro-expansion var-env form)))
    (compile-expression form1 var-env func-env initializer-p)))


;;;
;;; Literal
;;;

(defun compile-literal (form)
  (cond
    ((bool-literal-p form) (compile-bool-literal form))
    ((int-literal-p form) (compile-int-literal form))
    ((float-literal-p form) (compile-float-literal form))
    ((double-literal-p form) (compile-double-literal form))
    (t (error "The value ~S is an invalid expression." form))))

(defun compile-bool-literal (form)
  (compile-bool form))

(defun compile-int-literal (form)
  (compile-int form))

(defun compile-float-literal (form)
  (compile-float form))

(defun compile-double-literal (form)
  (compile-double form))


;;;
;;; CUDA dimension
;;;

(defun compile-cuda-dimension (form)
  (ecase form
    (grid-dim-x "gridDim.x")
    (grid-dim-y "gridDim.y")
    (grid-dim-z "gridDim.z")
    (block-idx-x "blockIdx.x")
    (block-idx-y "blockIdx.y")
    (block-idx-z "blockIdx.z")
    (block-dim-x "blockDim.x")
    (block-dim-y "blockDim.y")
    (block-dim-z "blockDim.z")
    (thread-idx-x "threadIdx.x")
    (thread-idx-y "threadIdx.y")
    (thread-idx-z "threadIdx.z")))


;;;
;;; Reference
;;;

(defun compile-reference (form var-env func-env)
  (cond
    ((variable-reference-p form)
     (compile-variable-reference form var-env))
    ((structure-reference-p form)
     (compile-structure-reference form var-env func-env))
    ((array-reference-p form)
     (compile-array-reference form var-env func-env))
    (t (error "The value ~S is an invalid form." form))))


;;;
;;; Reference - Variable
;;;

(defun compile-variable-reference (form var-env)
  (cond
    ((variable-environment-variable-exists-p var-env form)
     (compile-symbol form))
    ((variable-environment-global-exists-p var-env form)
     (variable-environment-global-c-name var-env form))
    (t
     (error "The variable ~S not found." form))))


;;;
;;; Reference - Structure
;;;

(defun compile-structure-reference (form var-env func-env)
  (let ((accessor (structure-reference-accessor form))
        (expr (structure-reference-expr form)))
    ;; check if the expression part of structure reference has the
    ;; same type as accessor's structure
    (let ((structure (structure-from-accessor accessor))
          (expr-type (type-of-expression expr var-env func-env)))
      (unless (eq structure expr-type)
        (error "The structure reference ~S is invalid." form)))
    (let ((accessor1 (structure-accessor-cuda-accessor accessor))
          (expr1 (compile-expression expr var-env func-env)))
      (format nil "~A.~A" expr1 accessor1))))


;;;
;;; Reference - Array
;;;

(defun compile-array-indices (indices var-env func-env)
  (mapcar #'(lambda (index)
              (compile-expression index var-env func-env))
          indices))

(defun compile-array-reference (form var-env func-env)
  (let ((expr (array-reference-expr form))
        (indices (array-reference-indices form)))
    ;; check if the expression part of array reference has the same
    ;; dimension as the array reference
    (let ((expr-type (type-of-expression expr var-env func-env)))
      (unless (= (array-type-dimension expr-type) (length indices))
        (error "The dimension of array reference ~S is invalid." form)))
    (let ((expr1 (compile-expression expr var-env func-env))
          (indices1 (compile-array-indices indices var-env func-env)))
      (format nil "~A~{[~A]~}" expr1 indices1))))


;;;
;;; Inline-if
;;;

(defun compile-inline-if (form var-env func-env)
  (let ((test-expr (inline-if-test-expression form))
        (then-expr (inline-if-then-expression form))
        (else-expr (inline-if-else-expression form)))
    ;; check if the test part of inline-if expression has bool type
    (let ((test-type (type-of-expression test-expr var-env func-env)))
      (unless (eq test-type 'bool)
        (error "The type of expression ~S is invalid." form)))
    ;; check if the then part of inline-of expression has the same
    ;; type as the else part of it
    (let ((then-type (type-of-expression then-expr var-env func-env))
          (else-type (type-of-expression else-expr var-env func-env)))
      (unless (eq then-type else-type)
        (error "The type of expression ~S is invalid." form)))
    (let ((test-expr1 (compile-expression test-expr var-env func-env))
          (then-expr1 (compile-expression then-expr var-env func-env))
          (else-expr1 (compile-expression else-expr var-env func-env)))
      (format nil "(~A ? ~A : ~A)" test-expr1 then-expr1 else-expr1))))


;;;
;;; Vector constructors
;;;

(defun compile-constructor (form var-env func-env initializer-p)
  (if initializer-p
      (let* ((operands (constructor-operands form))
             (operands1 (compile-operands operands var-env func-env)))
        (format nil "{ ~{~A~^, ~} }" operands1))
      ;; Delegete the logic to COMPILE-FUNCTION for historical reasons.
      (compile-function form var-env func-env)))


;;;
;;; Arithmetic operations
;;;

(defun compile-arithmetic (form var-env func-env)
  (let ((operator (arithmetic-operator form))
        (operands (arithmetic-operands form)))
    (if (<= (length operands) 2)
        (compile-function form var-env func-env)
        (let ((operand-head (butlast operands))
              (operand-tail (car (last operands))))
          (let ((form1 `(,operator (,operator ,@operand-head)
                                   ,operand-tail)))
            (compile-expression form1 var-env func-env))))))


;;;
;;; Function application
;;;

(defun type-of-operands (operands var-env func-env)
  (cl-cuda.lang.compiler.type-of-expression::type-of-operands operands var-env
                                                              func-env))

(defun compile-operands (operands var-env func-env)
  (mapcar #'(lambda (operand)
              (compile-expression operand var-env func-env))
          operands))

(defun compile-function (form var-env func-env)
  (let ((operator (function-operator form)))
    (if (function-environment-function-exists-p func-env operator)
        (compile-user-defined-function form var-env func-env)
        (compile-built-in-function form var-env func-env))))

(defun compile-user-defined-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    ;; check if the operands have the same types as the operator expect
    (let ((expected (function-environment-function-argument-types
                       func-env operator))
          (actual (type-of-operands operands var-env func-env)))
      (unless (equal expected actual)
        (error "The function application ~S is invalid." form)))
    (let ((operator1 (function-environment-function-c-name func-env
                                                           operator))
          (operands1 (compile-operands operands var-env func-env)))
      (if operands1
          (format nil "~A( ~{~A~^, ~} )" operator1 operands1)
          (format nil "~A()" operator1)))))

(defun compile-built-in-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((operand-types (type-of-operands operands var-env func-env)))
      (if (built-in-function-infix-p operator operand-types)
          (compile-built-in-infix-function operator operands operand-types
                                           var-env func-env)
          (compile-built-in-prefix-function operator operands operand-types
                                            var-env func-env)))))

(defun compile-built-in-infix-function (operator operands operand-types
                                        var-env func-env)
    (let ((op (built-in-function-c-name operator operand-types))
          (lhe (compile-expression (car operands) var-env func-env))
          (rhe (compile-expression (cadr operands) var-env func-env)))
      (format nil "(~A ~A ~A)" lhe op rhe)))

(defun compile-built-in-prefix-function (operator operands operand-types
                                         var-env func-env)
    (let ((operator1 (built-in-function-c-name operator operand-types))
          (operands1 (compile-operands operands var-env func-env)))
      (if operands1
          (format nil "~A( ~{~A~^, ~} )" operator1 operands1)
          (format nil "~A()" operator1))))
