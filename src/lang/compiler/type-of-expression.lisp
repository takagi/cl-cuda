#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.compiler.type-of-expression
  (:use :cl
        :cl-cuda.lang.type
        :cl-cuda.lang.syntax
        :cl-cuda.lang.environment
        :cl-cuda.lang.built-in)
  (:export :type-of-expression))
(in-package :cl-cuda.lang.compiler.type-of-expression)


;;;
;;; Type of expression
;;;

(defun type-of-expression (form var-env func-env)
  (cond
    ((%macro-p form func-env) (type-of-macro form var-env func-env))
    ((%symbol-macro-p form var-env)
     (type-of-symbol-macro form var-env func-env))
    ((literal-p form) (type-of-literal form))
    ((cuda-dimension-p form) (type-of-cuda-dimension form))
    ((reference-p form) (type-of-reference form var-env func-env))
    ((inline-if-p form) (type-of-inline-if form var-env func-env))
    ((arithmetic-p form) (type-of-arithmetic form var-env func-env))
    ((function-p form) (type-of-function form var-env func-env))
    (t (error "The value ~S is an invalid expression." form))))


;;;
;;; Macro
;;;

(defun %macro-p (form func-env)
  (and (macro-p form)
       (function-environment-macro-exists-p func-env
                                            (macro-operator form))))

(defun type-of-macro (form var-env func-env)
  (let ((name (macro-operator form))
        (arguments (macro-operands form)))
    (let ((expander (function-environment-macro-expander func-env name)))
      (let ((form1 (funcall expander arguments)))
        (type-of-expression form1 var-env func-env)))))


;;;
;;; Symbol macro
;;;

(defun %symbol-macro-p (form var-env)
  (and (symbol-macro-p form)
       (variable-environment-symbol-macro-exists-p var-env form)))

(defun type-of-symbol-macro (form var-env func-env)
  (let ((form1 (variable-environment-symbol-macro-expansion var-env form)))
    (type-of-expression form1 var-env func-env)))


;;;
;;; Literal
;;;

(defun type-of-literal (form)
  (cond
    ((bool-literal-p form) 'bool)
    ((int-literal-p form) 'int)
    ((float-literal-p form) 'float)
    ((double-literal-p form) 'double)
    (t (error "The value ~S is an invalid expression." form))))


;;;
;;; CUDA dimension
;;;

(defun type-of-cuda-dimension (form)
  (declare (ignore form))
  'int)


;;;
;;; Reference
;;;

(defun type-of-reference (form var-env func-env)
  (cond
    ((variable-reference-p form)
     (type-of-variable-reference form var-env))
    ((structure-reference-p form)
     (type-of-structure-reference form var-env func-env))
    ((array-reference-p form)
     (type-of-array-reference form var-env func-env))
    (t (error "The value ~S is an invalid expression." form))))


;;;
;;; Reference - Variable
;;;

(defun type-of-variable-reference (form var-env)
  (unless (variable-environment-variable-exists-p var-env form)
    (error "The variable ~S not found." form))
  (variable-environment-variable-type var-env form))


;;;
;;; Reference - Structure
;;;

(defun type-of-structure-reference (form var-env func-env)
  (let ((accessor (structure-reference-accessor form))
        (expr (structure-reference-expr form)))
    ;; check if the expression part of structure reference has the
    ;; same type as accessor's structure
    (let ((structure (structure-from-accessor accessor))
          (expr-type (type-of-expression expr var-env func-env)))
      (unless (eq structure expr-type)
        (error "The structure reference ~S is invalid." form)))
    (structure-accessor-return-type accessor)))


;;;
;;; Reference - Array
;;;

(defun type-of-array-reference (form var-env func-env)
  (let ((expr (array-reference-expr form))
        (indices (array-reference-indices form)))
    (let ((expr-type (type-of-expression expr var-env func-env)))
      ;; check if the expression part of array reference has the same
      ;; dimension as the array reference
      (unless (= (array-type-dimension expr-type) (length indices))
        (error "The dimension of array reference ~S is invalid." form))
      (array-type-base expr-type))))


;;;
;;; Inline-if
;;;

(defun type-of-inline-if (form var-env func-env)
  (let ((test-expr (inline-if-test-expression form))
        (then-expr (inline-if-then-expression form))
        (else-expr (inline-if-else-expression form)))
    ;; check if the test part of inline-if expression has bool type
    (let ((test-type (type-of-expression test-expr var-env func-env)))
      (unless (eq test-type 'bool)
        (error "The type of expression ~S is invalid." form)))
    (let ((then-type (type-of-expression then-expr var-env func-env))
          (else-type (type-of-expression else-expr var-env func-env)))
      ;; check if the then part of inline-of expression has the same
      ;; type as the else part of it
      (unless (eq then-type else-type)
        (error "The type of expression ~S is invalid." form))
      then-type)))


;;;
;;; Arithmetic operations
;;;

(defun type-of-arithmetic (form var-env func-env)
  (let ((operator (arithmetic-operator form))
        (operands (arithmetic-operands form)))
    (if (<= (length operands) 2)
        (type-of-function form var-env func-env)
        (let ((operand-head (car operands))
              (operand-tail (cdr operands)))
          (let ((form1 `(,operator ,operand-head
                                   (,operator ,@operand-tail))))
            (type-of-expression form1 var-env func-env))))))


;;;
;;; Function application
;;;

(defun type-of-operands (operands var-env func-env)
  (mapcar #'(lambda (operand)
              (type-of-expression operand var-env func-env))
          operands))

(defun type-of-function (form var-env func-env)
  (let ((operator (function-operator form)))
    (if (function-environment-function-exists-p func-env operator)
        (type-of-user-defined-function form var-env func-env)
        (type-of-built-in-function form var-env func-env))))

(defun type-of-user-defined-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    ;; check if the operands have the same types as the operator expect
    (let ((expected (function-environment-function-argument-types
                      func-env operator))
          (actual (type-of-operands operands var-env func-env)))
      (unless (equal expected actual)
        (error "The function application ~S is invalid." form)))
    (function-environment-function-return-type func-env operator)))

(defun type-of-built-in-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((argument-types (type-of-operands operands var-env func-env)))
      (built-in-function-return-type operator argument-types))))
