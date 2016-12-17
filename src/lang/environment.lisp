#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.environment
  (:use :cl
        :cl-cuda.lang.util
        :cl-cuda.lang.data
        :cl-cuda.lang.type)
  (:export ;; Variable environment
           :empty-variable-environment
           ;; Variable environment - Variable
           :variable-environment-add-variable
           :variable-environment-variable-exists-p
           :variable-environment-variable-name
           :variable-environment-variable-type
           ;; Variable environment - Symbol macro
           :variable-environment-add-symbol-macro
           :variable-environment-symbol-macro-exists-p
           :variable-environment-symbol-macro-name
           :variable-environment-symbol-macro-expansion
           ;; Variable environment - Global
           :variable-environment-add-global
           :variable-environment-global-exists-p
           :variable-environment-global-name
           :variable-environment-global-c-name
           :variable-environment-global-type
           :variable-environment-global-initializer
           ;; Function environment
           :empty-function-environment
           ;; Function environment - Function
           :function-environment-add-function
           :function-environment-function-exists-p
           :function-environment-function-name
           :function-environment-function-c-name
           :function-environment-function-return-type
           :function-environment-function-argument-types
           ;; Function environment - Macro
           :function-environment-add-macro
           :function-environment-macro-exists-p
           :function-environment-macro-name
           :function-environment-macro-expander)
  (:shadow :variable)
  (:import-from :alexandria
                :with-gensyms))
(in-package :cl-cuda.lang.environment)


;;;
;;; Variable environment
;;;

(defun empty-variable-environment ()
  nil)


;;;
;;; Variable environment - Variable
;;;

(defun variable-environment-add-variable (name type var-env)
  (let ((elem (make-variable name type)))
    (acons name elem var-env)))

(defun variable-environment-variable-exists-p (var-env name)
  (variable-p (cdr (assoc name var-env))))

(defun %lookup-variable (var-env name)
  (unless (variable-environment-variable-exists-p var-env name)
    (error "The variable ~S not found." name))
  (cdr (assoc name var-env)))

(defun variable-environment-variable-name (var-env name)
  (variable-name (%lookup-variable var-env name)))

(defun variable-environment-variable-type (var-env name)
  (variable-type (%lookup-variable var-env name)))


;;;
;;; Variable environment - Symbol macro
;;;

(defun variable-environment-add-symbol-macro (name expansion var-env)
  (let ((elem (make-symbol-macro name expansion)))
    (acons name elem var-env)))

(defun variable-environment-symbol-macro-exists-p (var-env name)
  (symbol-macro-p (cdr (assoc name var-env))))

(defun %lookup-symbol-macro (var-env name)
  (unless (variable-environment-symbol-macro-exists-p var-env name)
    (error "The symbol macro ~S not found." name))
  (cdr (assoc name var-env)))

(defun variable-environment-symbol-macro-name (var-env name)
  (symbol-macro-name (%lookup-symbol-macro var-env name)))

(defun variable-environment-symbol-macro-expansion (var-env name)
  (symbol-macro-expansion (%lookup-symbol-macro var-env name)))


;;;
;;; Variable environment - Global
;;;

(defun variable-environment-add-global (name type expression var-env)
  (check-type var-env list)
  (let ((elem (make-global name type expression)))
    (acons name elem var-env)))

(defun variable-environment-global-exists-p (var-env name)
  (check-type name cl-cuda-symbol)
  (global-p (cdr (assoc name var-env))))

(defun %lookup-global (var-env name)
  (unless (variable-environment-global-exists-p var-env name)
    (error "The variable ~S not found." name))
  (cdr (assoc name var-env)))

(defun variable-environment-global-name (var-env name)
  (global-name (%lookup-global var-env name)))

(defun variable-environment-global-c-name (var-env name)
  (global-c-name (%lookup-global var-env name)))

(defun variable-environment-global-type (var-env name)
  (global-type (%lookup-global var-env name)))

(defun variable-environment-global-initializer (var-env name)
  (global-initializer (%lookup-global var-env name)))


;;;
;;; Function environment
;;;

(defun empty-function-environment ()
  '())


;;;
;;; Function environment - Function
;;;

(defun function-environment-add-function (name return-type
                                          argument-types func-env)
  (let ((elem (make-function name return-type argument-types)))
    (acons name elem func-env)))

(defun function-environment-function-exists-p (func-env name)
  (function-p (cdr (assoc name func-env))))

(defun %lookup-function (func-env name)
  (unless (function-environment-function-exists-p func-env name)
    (error "The function ~S is undefined." name))
  (cdr (assoc name func-env)))

(defun function-environment-function-name (func-env name)
  (function-name (%lookup-function func-env name)))

(defun function-environment-function-c-name (func-env name)
  (function-c-name (%lookup-function func-env name)))

(defun function-environment-function-return-type (func-env name)
  (function-return-type (%lookup-function func-env name)))

(defun function-environment-function-argument-types (func-env name)
  (function-argument-types (%lookup-function func-env name)))


;;;
;;; Function environment - Macro
;;;

(defun function-environment-add-macro (name arguments body func-env)
  (let ((elem (make-macro name arguments body)))
    (acons name elem func-env)))

(defun function-environment-macro-exists-p (func-env name)
  (macro-p (cdr (assoc name func-env))))

(defun %lookup-macro (func-env name)
  (unless (function-environment-macro-exists-p func-env name)
    (error "The macro ~S is undefined." name))
  (cdr (assoc name func-env)))

(defun function-environment-macro-name (func-env name)
  (macro-name (%lookup-macro func-env name)))

(defun function-environment-macro-expander (func-env name)
  (macro-expander (%lookup-macro func-env name)))


;;;
;;; Variable
;;;

(defstruct (variable (:constructor %make-variable))
  (name :name :read-only t)
  (type :type :read-only t))

(defun make-variable (name type)
  (unless (cl-cuda-symbol-p name)
    (error 'type-error :datum name :expected-type 'cl-cuda-symbol))
  (unless (cl-cuda-type-p type)
    (error 'type-error :datum type :expected-type 'cl-cuda-type))
  (%make-variable :name name :type type))


;;;
;;; Symbol macro
;;;

(defstruct (symbol-macro (:constructor %make-symbol-macro))
  (name :name :read-only t)
  (expansion :expansion :read-only t))

(defun make-symbol-macro (name expansion)
  (unless (cl-cuda-symbol-p name)
    (error 'type-error :datum name :expected-type 'cl-cuda-symbol))
  (%make-symbol-macro :name name :expansion expansion))


;;;
;;; Global
;;;

(defstruct (global (:constructor %make-global))
  (name :name :read-only t)
  (type :type :read-only t)
  (initializer :initializer :read-only t))

(defun make-global (name type initializer)
  (check-type name cl-cuda-symbol)
  (check-type type cl-cuda-type)
  (%make-global :name name :type type :initializer initializer))

(defun global-c-name (global)
  (c-identifier (global-name global) t))


;;;
;;; Function
;;;

;; use name begining with '%' to avoid package locking
(defstruct (%function (:constructor %make-function)
                      (:conc-name function-)
                      (:predicate function-p))
  (name :name :read-only t)
  (return-type :return-type :read-only t)
  (argument-types :argument-types :read-only t))

(defun make-function (name return-type argument-types)
  (unless (cl-cuda-symbol-p name)
    (error 'type-error :datum name :expected-type 'cl-cuda-symbol))
  (unless (cl-cuda-type-p return-type)
    (error 'type-error :datum return-type :expected-type 'cl-cuda-type))
  (dolist (argument-type argument-types)
    (unless (cl-cuda-type-p argument-type)
      (error 'type-error :datum argument-type
                         :expected-type 'cl-cuda-type)))
  (%make-function :name name
                  :return-type return-type
                  :argument-types argument-types))

(defun function-c-name (function)
  (c-identifier (function-name function) t))


;;;
;;; Macro
;;;

(defstruct (macro (:constructor %make-macro))
  (name :name :read-only t)
  (arguments :arguments :read-only t)
  (body :body :read-only t))

(defun make-macro (name arguments body)
  (unless (cl-cuda-symbol-p name)
    (error 'type-error :datum name :expected-type 'cl-cuda-symbol))
  (unless (listp arguments)
    (error 'type-error :datum arguments :expected-type 'list))
  (unless (listp body)
    (error 'type-error :datum body :expected-type 'list))
  (%make-macro :name name :arguments arguments :body body))

(defun macro-expander (macro)
  (let ((arguments (macro-arguments macro))
        (body (macro-body macro)))
    (with-gensyms (arguments1)
      (eval `#'(lambda (,arguments1)
                 (destructuring-bind ,arguments ,arguments1
                   ,@body))))))
