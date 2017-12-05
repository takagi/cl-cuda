#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.lang.compiler.compile-statement
  (:use :cl
        :cl-cuda.lang.util
        :cl-cuda.lang.type
        :cl-cuda.lang.syntax
        :cl-cuda.lang.environment
        :cl-cuda.lang.compiler.compile-data
        :cl-cuda.lang.compiler.compile-type
        :cl-cuda.lang.compiler.type-of-expression
        :cl-cuda.lang.compiler.compile-expression)
  (:export :compile-statement))
(in-package :cl-cuda.lang.compiler.compile-statement)


;;;
;;; Compile statement
;;;

(defun compile-statement (form var-env func-env)
  (cond
    ((%macro-p form func-env) (compile-macro form var-env func-env))
    ((if-p form) (compile-if form var-env func-env))
    ((let-p form) (compile-let form var-env func-env))
    ((symbol-macrolet-p form) (compile-symbol-macrolet form var-env func-env))
    ((macrolet-p form) (compile-macrolet form var-env func-env))
    ((do-p form) (compile-do form var-env func-env))
    ((with-shared-memory-p form)
     (compile-with-shared-memory form var-env func-env))
    ((set-p form) (compile-set form var-env func-env))
    ((progn-p form) (compile-progn form var-env func-env))
    ((return-p form) (compile-return form var-env func-env))
    ((function-p form) (compile-function form var-env func-env))
    (t (error "The value ~S is an invalid statement." form))))


;;;
;;; Macro
;;;

(defun %macro-p (form func-env)
  (cl-cuda.lang.compiler.compile-expression::%macro-p form func-env))

(defun compile-macro (form var-env func-env)
  (let ((operator (macro-operator form))
        (operands (macro-operands form)))
    (let ((expander (function-environment-macro-expander func-env operator)))
      (let ((form1 (funcall expander operands)))
        (compile-statement form1 var-env func-env)))))


;;;
;;; If statement
;;;

(defun compile-if (form var-env func-env)
  (let ((test-expr (if-test-expression form))
        (then-stmt (if-then-statement form))
        (else-stmt (if-else-statement form)))
    ;; check if the test part of inline-if expression has bool type
    (let ((test-type (type-of-expression test-expr var-env func-env)))
      (unless (eq test-type 'bool)
        (error "The type of statement ~S is invalid." form)))
    (let ((test-expr1 (compile-expression test-expr var-env func-env))
          (then-stmt1 (compile-statement then-stmt var-env func-env))
          (else-stmt1 (if else-stmt
                          (compile-statement else-stmt var-env func-env))))
      (let ((then-stmt2 (indent 2 then-stmt1))
            (else-stmt2 (if else-stmt1
                            (indent 2 else-stmt1))))
        (format nil "if (~A) {~%~A}~@[ else {~%~A}~]~%" test-expr1
                                                        then-stmt2
                                                        else-stmt2)))))


;;;
;;; Let statement
;;;

(defun var-env-add-let-bindings (var-env func-env bindings)
  (flet ((aux (var-env0 binding)
           (let* ((var (let-binding-var binding))
                  (expr (let-binding-expr binding))
                  (type (type-of-expression expr var-env func-env)))
             (variable-environment-add-variable var type var-env0))))
    (reduce #'aux bindings :initial-value var-env)))

(defun compile-let-bindings (bindings var-env func-env)
  (flet ((aux (binding)
           (let* ((var (let-binding-var binding))
                  (expr (let-binding-expr binding))
                  (type (type-of-expression expr var-env func-env)))
             (let ((var1 (compile-symbol var))
                   (expr1 (compile-expression expr var-env func-env))
                   (type1 (compile-type type)))
               (format nil "~A ~A = ~A;~%" type1 var1 expr1)))))
    (format nil "~{~A~}" (mapcar #'aux bindings))))

(defun compile-let-statements (statements var-env func-env)
  (compile-statement `(progn ,@statements) var-env func-env))

(defun compile-let (form var-env func-env)
  (let ((bindings (let-bindings form))
        (statements (let-statements form)))
    (let ((var-env1 (var-env-add-let-bindings var-env func-env bindings)))
      (let ((bindings1 (compile-let-bindings bindings var-env func-env))
            (statements1 (compile-let-statements statements var-env1
                                                            func-env)))
        (let ((bindings2 (indent 2 bindings1))
              (statements2 (indent 2 statements1)))
          (format nil "{~%~A~A}~%" bindings2 statements2))))))


;;;
;;; Symbol-macrolet statement
;;;

(defun var-env-add-symbol-macrolet-bindings (var-env bindings)
  (flet ((aux (var-env0 binding)
           (let* ((symbol (symbol-macrolet-binding-symbol binding))
                  (expansion (symbol-macrolet-binding-expansion binding)))
             (variable-environment-add-symbol-macro symbol expansion
                                                    var-env0))))
    (reduce #'aux bindings :initial-value var-env)))

(defun compile-symbol-macrolet-statements (statements var-env func-env)
  (compile-statement `(progn ,@statements) var-env func-env))

(defun compile-symbol-macrolet (form var-env func-env)
  (let ((bindings (symbol-macrolet-bindings form))
        (statements (symbol-macrolet-statements form)))
    (let ((var-env1 (var-env-add-symbol-macrolet-bindings var-env
                                                          bindings)))
      (let ((statements1 (compile-symbol-macrolet-statements statements
                                                             var-env1
                                                             func-env)))
        (let ((statements2 (indent 2 statements1)))
          (format nil "{~%~A}~%" statements2))))))


;;;
;;; Symbol-macrolet statement
;;;

(defun func-env-add-macrolet-bindings (func-env bindings)
  (flet ((aux (func-env0 binding)
           (let ((symbol (macrolet-binding-symbol binding))
                 (arguments (macrolet-binding-arguments binding))
                 (body (macrolet-binding-body binding)))
             (function-environment-add-macro symbol arguments body
                                             func-env0))))
    (reduce #'aux bindings :initial-value func-env)))

(defun compile-macrolet-statements (statements var-env func-env)
  (compile-statement `(progn ,@statements) var-env func-env))

(defun compile-macrolet (form var-env func-env)
  (let ((bindings (macrolet-bindings form))
        (statements (macrolet-statements form)))
    (let ((func-env1 (func-env-add-macrolet-bindings func-env bindings)))
      (let ((statements1 (compile-macrolet-statements statements
                                                      var-env
                                                      func-env1)))
        (let ((statements2 (indent 2 statements1)))
          (format nil "{~%~A}~%" statements2))))))


;;;
;;; Do statement
;;;

(defun var-env-add-do-bindings (var-env func-env bindings)
  (flet ((aux (var-env0 binding)
           (let* ((var (do-binding-var binding))
                  (init (do-binding-init binding))
                  (type (type-of-expression init var-env func-env)))
             (variable-environment-add-variable var type var-env0))))
    (reduce #'aux bindings :initial-value var-env)))

(defun compile-do-init-part (bindings var-env func-env)
  (flet ((aux (binding)
           (let* ((var (do-binding-var binding))
                  (init (do-binding-init binding))
                  (type (type-of-expression init var-env func-env)))
             (let ((var1 (compile-symbol var))
                   (init1 (compile-expression init var-env func-env))
                   (type1 (compile-type type)))
               (format nil "~A ~A = ~A" type1 var1 init1)))))
    (format nil "~{~A~^, ~}" (mapcar #'aux bindings))))

(defun compile-do-test-part (end-test var-env func-env)
  (let ((end-test1 (compile-expression end-test var-env func-env)))
    (format nil "! ~A" end-test1)))

(defun compile-do-step-part (bindings var-env func-env)
  (flet ((aux (binding)
           (let ((var (do-binding-var binding))
                 (step (do-binding-step binding)))
             (let ((var1 (compile-symbol var))
                   (step1 (compile-expression step var-env func-env)))
               (format nil "~A = ~A" var1 step1)))))
    (format nil "~{~A~^, ~}" (mapcar #'aux bindings))))

(defun compile-do-statements (statements var-env func-env)
  (compile-statement `(progn ,@statements) var-env func-env))

(defun compile-do (form var-env func-env)
  (let ((bindings (do-bindings form))
        (end-test (do-end-test form))
        (statements (do-statements form)))
    (let ((var-env1 (var-env-add-do-bindings var-env func-env bindings)))
      (let ((init-part (compile-do-init-part bindings var-env func-env))
            (test-part (compile-do-test-part end-test var-env1 func-env))
            (step-part (compile-do-step-part bindings var-env1 func-env))
            (statements1 (compile-do-statements statements var-env1
                                                           func-env)))
        (let ((statements2 (indent 2 statements1)))
          (format nil "for ( ~A; ~A; ~A )~%{~%~A}~%"
                      init-part test-part step-part statements2))))))


;;;
;;; With-shared-memory statement
;;;

(defun var-env-add-with-shared-memory-specs (var-env specs)
  (flet ((aux (var-env0 spec)
           (let* ((var (with-shared-memory-spec-var spec))
                  (type (with-shared-memory-spec-type spec))
                  (dims (length (with-shared-memory-spec-dimensions spec))))
             (let ((type1 (array-type type dims)))
               (variable-environment-add-variable var type1 var-env0)))))
    (reduce #'aux specs :initial-value var-env)))

(defun compile-with-shared-memory-spec-dimensions (dims var-env func-env)
  (flet ((aux (dim)
           (compile-expression dim var-env func-env)))
    (mapcar #'aux dims)))

(defun compile-with-shared-memory-specs (specs var-env func-env)
  (flet ((aux (spec)
           (let ((var (with-shared-memory-spec-var spec))
                 (type (with-shared-memory-spec-type spec))
                 (dims (with-shared-memory-spec-dimensions spec)))
             (let ((var1 (compile-symbol var))
                   (type1 (compile-type type))
                   (dims1 (compile-with-shared-memory-spec-dimensions
                            dims var-env func-env)))
               (format nil "__shared__ ~A ~A~{[~A]~};~%" type1 var1 dims1)))))
    (format nil "~{~A~}" (mapcar #'aux specs))))

(defun compile-with-shared-memory-statements (statements var-env func-env)
  (compile-statement `(progn ,@statements) var-env func-env))

(defun compile-with-shared-memory (form var-env func-env)
  (let ((specs (with-shared-memory-specs form))
        (statements (with-shared-memory-statements form)))
    (let ((var-env1 (var-env-add-with-shared-memory-specs var-env specs)))
      (let ((specs1 (compile-with-shared-memory-specs specs var-env func-env))
            (statements1 (compile-with-shared-memory-statements statements
                                                                var-env1
                                                                func-env)))
        (let ((specs2 (indent 2 specs1))
              (statements2 (indent 2 statements1)))
          (format nil "{~%~A~A}~%" specs2 statements2))))))


;;;
;;; Set statement
;;;

(defun compile-set (form var-env func-env)
  (let ((reference (set-reference form))
        (expr (set-expression form)))
    ;; check if the reference part of set statement has the same type
    ;; as the expression part of that
    (let ((ref-type (type-of-expression reference var-env func-env))
          (expr-type (type-of-expression expr var-env func-env)))
      (unless (eq ref-type expr-type)
        (error "The type of statement ~S is invalid." form)))
    (let ((reference1 (compile-expression reference var-env func-env))
          (expr1 (compile-expression expr var-env func-env)))
      (format nil "~A = ~A;~%" reference1 expr1))))


;;;
;;; Progn statement
;;;

(defun compile-progn (form var-env func-env)
  (flet ((aux (statement)
           (compile-statement statement var-env func-env)))
    (let ((statements (progn-statements form)))
      (let ((statements1 (mapcar #'aux statements)))
        (format nil "~{~A~}" statements1)))))


;;;
;;; Return statement
;;;

(defun compile-return (form var-env func-env)
  (let ((expr (return-expr form)))
    (if expr
        (let ((expr1 (compile-expression expr var-env func-env)))
          (format nil "return ~A;~%" expr1))
        (format nil "return;~%"))))


;;;
;;; Function application
;;;

(defun compile-function (form var-env func-env)
  (let ((code (cl-cuda.lang.compiler.compile-expression::compile-function
                form var-env func-env)))
    (format nil "~A;~%" code)))
