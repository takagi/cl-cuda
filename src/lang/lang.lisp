#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda.lang)


;;;
;;; Float3
;;;

(defstruct (float3 (:constructor make-float3 (x y z)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float))

(defun float3-= (a b)
  (and (= (float3-x a) (float3-x b))
       (= (float3-y a) (float3-y b))
       (= (float3-z a) (float3-z b))))

(cffi:defcstruct (float3 :class float3-c)
  (x :float)
  (y :float)
  (z :float))

(defmethod cffi:translate-into-foreign-memory ((value float3)
                                               (type float3-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z) ptr (:struct float3))
    (setf x (float3-x value)
          y (float3-y value)
          z (float3-z value))))

(defmethod cffi:translate-from-foreign (value (type float3-c))
  (cffi:with-foreign-slots ((x y z) value (:struct float3))
    (make-float3 x y z)))


;;;
;;; Float4
;;;

(defstruct (float4 (:constructor make-float4 (x y z w)))
  (x 0.0 :type single-float)
  (y 0.0 :type single-float)
  (z 0.0 :type single-float)
  (w 0.0 :type single-float))

(defun float4-= (a b)
  (and (= (float4-x a) (float4-x b))
       (= (float4-y a) (float4-y b))
       (= (float4-z a) (float4-z b))
       (= (float4-w a) (float4-w b))))

(cffi:defcstruct float4
  (x :float)
  (y :float)
  (z :float)
  (w :float))

(defmethod cffi:translate-into-foreign-memory ((value float4)
                                               (type float4-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z w) ptr (:struct float4))
    (setf x (float4-x value)
          y (float4-y value)
          z (float4-z value)
          w (float4-w value))))

(defmethod cffi:translate-from-foreign (value (type float3-c))
  (cffi:with-foreign-slots ((x y z w) value (:struct float4))
    (make-float4 x y z w)))


;;;
;;; Double3
;;;

(defstruct (double3 (:constructor make-double3 (x y z)))
  (x 0.0d0 :type double-float)
  (y 0.0d0 :type double-float)
  (z 0.0d0 :type double-float))

(defun double3-= (a b)
  (and (= (double3-x a) (double3-x b))
       (= (double3-y a) (double3-y b))
       (= (double3-z a) (double3-z b))))

(cffi:defcstruct double3
  (x :double)
  (y :double)
  (z :double))

(defmethod cffi:translate-into-foreign-memory ((value double3)
                                               (type double3-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z) ptr (:struct double3))
    (setf x (double3-x value)
          y (double3-y value)
          z (double3-z value))))

(defmethod cffi:translate-from-foreign (value (type double3-c))
  (cffi:with-foreign-slots ((x y z) value (:struct double3))
    (make-double3 x y z)))


;;;
;;; Double4
;;;

(defstruct (double4 (:constructor make-double4 (x y z w)))
  (x 0.0d0 :type double-float)
  (y 0.0d0 :type double-float)
  (z 0.0d0 :type double-float)
  (w 0.0d0 :type double-float))

(defun double4-= (a b)
  (and (= (double4-x a) (double4-x b))
       (= (double4-y a) (double4-y b))
       (= (double4-z a) (double4-z b))
       (= (double4-w a) (double4-w b))))

(cffi:defcstruct double4
  (x :double)
  (y :double)
  (z :double)
  (w :double))

(defmethod cffi:translate-into-foreign-memory ((value double4)
                                               (type double4-c)
                                               ptr)
  (cffi:with-foreign-slots ((x y z w) ptr (:struct double4))
    (setf x (double4-x value)
          y (double4-y value)
          z (double4-z value)
          w (double4-w value))))

(defmethod cffi:translate-from-foreign (value (type double3-c))
  (cffi:with-foreign-slots ((x y z w) value (:struct double4))
    (make-double4 x y z w)))


;;;
;;; CURAND
;;;

(cffi:defcstruct curand-state-xorwow
  (d :unsigned-int)
  (v :unsigned-int :count 5)
  (boxmuller-flag :int)
  (boxmuller-flag-double :int)
  (boxmuller-extra :float)
  (boxmuller-extra-double :double))



;;;
;;; Types
;;;

(defparameter +basic-types+ `((void  0 :void)
                              (bool  1 (:boolean :int8))
                              (int   4 :int)
                              (float 4 :float)
                              (double 8 :double)
                              (curand-state-xorwow
                               ,(cffi:foreign-type-size
                                 '(:struct curand-state-xorwow))
                               (:pointer :struct curand-state-xorwow))))

(defvar +vector-types+ '((float3 float 3 float3-x float3-y float3-z)
                         (float4 float 4 float4-x float4-y float4-z float4-w)
                         (double3 double 3 double3-x double3-y double3-z)
                         (double4 double 4 double4-x double4-y double4-z double4-w)))

(defvar +vector-type-elements+ '(x y z w))

(defun cffi-type-size (type)
  (type-size type))

(defun type-size (type)
  (cond
    ((basic-type-p type)  (basic-type-size type))
    ((vector-type-p type) (vector-type-size type))
    ((array-type-p type)  (array-type-pointer-size type))
    (t (error "invalid type:~A" type))))

(defun valid-type-p (type)
  (or (basic-type-p  type)
      (vector-type-p type)
      (array-type-p  type)))

(defun cffi-type (type)
  (cond
    ((basic-type-p  type) (basic-cffi-type  type))
    ((vector-type-p type) (vector-cffi-type type))
    ((array-type-p  type) (array-cffi-type  type))
    (t (error "invalid type: ~A" type))))

(defun non-pointer-type-p (type)
  (or (basic-type-p type)
      (vector-type-p type)))

(defun basic-type-size (type)
  (or (cadr (assoc type +basic-types+))
      (error "invalid type: ~A" type)))

(defun basic-type-p (type)
  (and (assoc type +basic-types+)
       t))

(defun basic-cffi-type (type)
  (or (caddr (assoc type +basic-types+))
      (error "invalid type: ~A" type)))

(defun vector-type-size (type)
  (* (vector-type-length type)
     (type-size (vector-type-base-type type))))

(defun vector-type-p (type)
  (and (assoc type +vector-types+)
       t))

(defun vector-cffi-type (type)
  (unless (vector-type-p type)
    (error "invalid type: ~A" type))
  (list :struct type))

(defun vector-types ()
  (mapcar #'car +vector-types+))

(defun vector-type-base-type (type)
  (or (cadr (assoc type +vector-types+))
      (error "invalid type: ~A" type)))

(defun vector-type-length (type)
  (or (caddr (assoc type +vector-types+))
      (error "invalid type: ~A" type)))

(defun vector-type-elements (type)
  (loop repeat (vector-type-length type)
     for elm in +vector-type-elements+
     collect elm))

(defun vector-type-selectors (type)
  (or (cdddr (assoc type +vector-types+))
      (error "invalid type: ~A" type)))

(defun valid-vector-type-selector-p (selector)
  (let ((selectors (apply #'append (mapcar #'vector-type-selectors (vector-types)))))
    (and (find selector selectors)
         t)))

(defun vector-type-selector-type (selector)
  (loop for type in (vector-types)
     when (member selector (vector-type-selectors type))
     return type
     finally (error "invalid selector: ~A" selector)))

(defun array-type-p (type)
  (let ((type-str (symbol-name type)))
    (let ((last (aref type-str (1- (length type-str))))
          (rest (remove-star type)))
      (and (eq last #\*)
           (or (basic-type-p rest) (vector-type-p rest))))))

(defun array-cffi-type (type)
  (unless (array-type-p type)
    (error (format nil "invalid type: ~A" type)))
  'cu-device-ptr)

(defun array-type-pointer-size (type)
  (unless (array-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (cffi:foreign-type-size 'cu-device-ptr))

(defun array-type-dimension (type)
  (unless (array-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (count #\* (princ-to-string type)))

(defun add-star (type n)
  (labels ((aux (str n2)
             (if (< n2 1)
                 str
                 (aux (concatenate 'string str "*") (1- n2)))))
    (cl-cuda-symbolicate (aux (princ-to-string type) n))))

(defun remove-star (type)
  (let ((rev (reverse (princ-to-string type))))
    (if (string= (subseq rev 0 1) "*")
        (remove-star (cl-cuda-symbolicate (reverse (subseq rev 1))))
        type)))

;;;
;;; Definition of kernel definition
;;;

(defun make-kerdef-function (name return-type args body)
  (assert (symbolp name))
  (assert (valid-type-p return-type))
  (dolist (arg args)
    (assert (= (length arg) 2))
    (assert (symbolp (car arg)))
    (assert (valid-type-p (cadr arg))))
  (assert (listp body))
  (list name :function return-type args body))

(defun kerdef-function-p (elem)
  (cl-pattern:match elem
    ((_ :function _ _ _) t)
    (_ nil)))

(defun kerdef-function-name (elem)
  (cl-pattern:match elem
    ((name :function _ _ _) name)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun kerdef-function-c-name (elem)
  (compile-identifier-with-package-name (kerdef-function-name elem)))

(defun kerdef-function-return-type (elem)
  (cl-pattern:match elem
    ((_ :function return-type _ _) return-type)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun kerdef-function-arguments (elem)
  (cl-pattern:match elem
    ((_ :function _ args _) args)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun kerdef-function-argument-types (elem)
  (mapcar #'cadr (kerdef-function-arguments elem)))

(defun kerdef-function-body (elem)
  (cl-pattern:match elem
    ((_ :function _ _ body) body)
    (_ (error "invalid kernel definition function: ~A" elem))))

(defun make-kerdef-macro (name args body expander)
  (assert (symbolp name))
  (assert (listp args))
  (assert (listp body))
  (assert (functionp expander))
  (list name :macro args body expander))

(defun kerdef-macro-p (elem)
  (cl-pattern:match elem
    ((_ :macro _ _ _) t)
    (_ nil)))

(defun kerdef-macro-name (elem)
  (cl-pattern:match elem
    ((name :macro _ _ _) name)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun kerdef-macro-arguments (elem)
  (cl-pattern:match elem
    ((_ :macro args _ _) args)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun kerdef-macro-body (elem)
  (cl-pattern:match elem
    ((_ :macro _ body _) body)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun kerdef-macro-expander (elem)
  (cl-pattern:match elem
    ((_ :macro _ _ expander) expander)
    (_ (error "invalid kernel definition macro: ~A" elem))))

(defun make-kerdef-constant (name type expression)
  (assert (symbolp name))
  (assert (valid-type-p type))
  (list name :constant type expression))

(defun kerdef-constant-p (elem)
  (cl-pattern:match elem
    ((_ :constant _ _) t)
    (_ nil)))

(defun kerdef-constant-name (elem)
  (cl-pattern:match elem
    ((name :constant _ _) name)
    (_ (error "invalid kernel definition constant: ~A" elem))))

(defun kerdef-constant-type (elem)
  (cl-pattern:match elem
    ((_ :constant type _) type)
    (_ (error "invalid kernel definition constant: ~A" elem))))

(defun kerdef-constant-expression (elem)
  (cl-pattern:match elem
    ((_ :constant _ exp) exp)
    (_ (error "invalid kernel definition constant: ~A" elem))))

(defun make-kerdef-symbol-macro (name expansion)
  (assert (symbolp name))
  (list name :symbol-macro expansion))

(defun kerdef-symbol-macro-p (elem)
  (cl-pattern:match elem
    ((_ :symbol-macro _) t)
    (_ nil)))

(defun kerdef-symbol-macro-name (elem)
  (cl-pattern:match elem
    ((name :symbol-macro _) name)
    (_ nil)))

(defun kerdef-symbol-macro-expansion (elem)
  (cl-pattern:match elem
    ((_ :symbol-macro expansion) expansion)
    (_ (error "invalid kernel definition symbol macro: ~A" elem))))

(defun kerdef-name (elem)
  (cl-pattern:match elem
    ((name :function _ _ _) name)
    ((name :macro _ _ _) name)
    ((name :constant _ _) name)
    ((name :symbol-macro _) name)
    (_ (error "invalid kernel definition element: ~A" elem))))

(defun empty-kernel-definition ()
  (list nil nil))

(defun add-function-to-kernel-definition (name return-type arguments body def)
  (destructuring-bind (func-table var-table) def
    (let ((elem (make-kerdef-function name return-type arguments body)))
      (list (remove-duplicates (cons elem func-table) :key #'kerdef-name :from-end t)
            var-table))))

(defun remove-function-from-kernel-definition (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list (remove name func-table :key #'kerdef-name) var-table)))

(defun add-macro-to-kernel-definition (name arguments body expander def)
  (destructuring-bind (func-table var-table) def
    (let ((elem (make-kerdef-macro name arguments body expander)))
      (list (remove-duplicates (cons elem func-table) :key #'kerdef-name :from-end t)
            var-table))))

(defun remove-macro-from-kernel-definition (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list (remove name func-table :key #'kerdef-name) var-table)))

(defun add-constant-to-kernel-definition (name type expression def)
  (destructuring-bind (func-table var-table) def
    (let ((elem (make-kerdef-constant name type expression)))
      (list func-table
            (remove-duplicates (cons elem var-table) :key #'kerdef-name :from-end t)))))

(defun remove-constant-from-kernel-definition (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list func-table (remove name var-table :key #'kerdef-name))))

(defun add-symbol-macro-to-kernel-definition (name expansion def)
  (destructuring-bind (funct-table var-table) def
    (let ((elem (make-kerdef-symbol-macro name expansion)))
      (list funct-table
            (remove-duplicates (cons elem var-table) :key #'kerdef-name :from-end t)))))

(defun remove-symbol-macro-from-kernel-definition (name def)
  (unless (kernel-definition-symbol-macro-exists-p name def)
    (error "undefined kernel definition symbol macro: ~A" name))
  (destructuring-bind (func-table var-table) def
    (list func-table (remove name var-table :key #'kerdef-name))))

(defun bulk-add-kernel-definition (bindings def)
  (reduce #'(lambda (def2 binding)
              (cl-pattern:match binding
                ((name :function return-type args body)
                 (add-function-to-kernel-definition name return-type args body def2))
                ((name :macro args body expander)
                 (add-macro-to-kernel-definition name args body expander def2))
                ((name :constant type exp)
                 (add-constant-to-kernel-definition name type exp def2))
                ((name :symbol-macro expansion)
                 (add-symbol-macro-to-kernel-definition name expansion def2))
                (_ (error "invalid kernel definition element: ~A" binding))))
          bindings :initial-value def))

(defmacro with-kernel-definition ((def bindings) &body body)
  (labels ((aux (binding)
             (cl-pattern:match binding
               ((name :function return-type args body) `(list ',name :function ',return-type ',args ',body))
               ((name :macro args body) (alexandria:with-gensyms (args0)
                                          `(list ',name :macro ',args ',body
                                                 (lambda (,args0) (destructuring-bind ,args ,args0 ,@body)))))
               ((name :constant type exp) `(list ',name :constant ',type ',exp))
               ((name :symbol-macro expansion) `(list ',name :symbol-macro ',expansion))
               (_ `',binding))))
    (let ((bindings2 `(list ,@(mapcar #'aux bindings))))
      `(let ((,def (bulk-add-kernel-definition ,bindings2 (empty-kernel-definition))))
         ,@body))))

(defun lookup-kernel-definition (kind name def)
  (destructuring-bind (func-table var-table) def
    (ecase kind
      (:function (let ((elem (find name func-table :key #'kerdef-name)))
                   (when (kerdef-function-p elem)
                     elem)))
      (:macro (let ((elem (find name func-table :key #'kerdef-name)))
                (when (kerdef-macro-p elem)
                  elem)))
      (:constant (let ((elem (find name var-table :key #'kerdef-name)))
                   (when (kerdef-constant-p elem)
                     elem)))
      (:symbol-macro (let ((elem (find name var-table :key #'kerdef-name)))
                       (when (kerdef-symbol-macro-p elem)
                         elem))))))

(defun kernel-definition-function-exists-p (name def)
  (and (lookup-kernel-definition :function name def)
       t))

(defun kernel-definition-macro-exists-p (name def)
  (and (lookup-kernel-definition :macro name def)
       t))

(defun kernel-definition-constant-exists-p (name def)
  (and (lookup-kernel-definition :constant name def)
       t))

(defun kernel-definition-symbol-macro-exists-p (name def)
  (and (lookup-kernel-definition :symbol-macro name def)
       t))

(defun kernel-definition-function-name (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-name (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-c-name (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-c-name (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-names (def)
  (destructuring-bind (func-table _) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-function-p func-table))))

(defun kernel-definition-function-return-type (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-return-type (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-arguments (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-arguments (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-argument-types (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-argument-types (lookup-kernel-definition :function name def)))

(defun kernel-definition-function-body (name def)
  (unless (kernel-definition-function-exists-p name def)
    (error "undefined kernel definition function: ~A" name))
  (kerdef-function-body (lookup-kernel-definition :function name def)))

(defun kernel-definition-macro-name (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-name (lookup-kernel-definition :macro name def)))

(defun kernel-definition-macro-names (def)
  (destructuring-bind (func-table _) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-macro-p func-table))))

(defun kernel-definition-macro-arguments (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-arguments (lookup-kernel-definition :macro name def)))

(defun kernel-definition-macro-body (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-body (lookup-kernel-definition :macro name def)))

(defun kernel-definition-macro-expander (name def)
  (unless (kernel-definition-macro-exists-p name def)
    (error "undefined kernel definition macro: ~A" name))
  (kerdef-macro-expander (lookup-kernel-definition :macro name def)))

(defun kernel-definition-constant-name (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (kerdef-constant-name (lookup-kernel-definition :constant name def)))

(defun kernel-definition-constant-names (def)
  (destructuring-bind (_ var-table) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-constant-p var-table))))

(defun kernel-definition-constant-type (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (kerdef-constant-type (lookup-kernel-definition :constant name def)))

(defun kernel-definition-constant-expression (name def)
  (unless (kernel-definition-constant-exists-p name def)
    (error "undefined kernel definition constant: ~A" name))
  (kerdef-constant-expression (lookup-kernel-definition :constant name def)))

(defun kernel-definition-symbol-macro-name (name def)
  (unless (kernel-definition-symbol-macro-exists-p name def)
    (error "undefined kernel definition symbol macro: ~A" name))
  (kerdef-symbol-macro-name (lookup-kernel-definition :symbol-macro name def)))

(defun kernel-definition-symbol-macro-names (def)
  (destructuring-bind (_ var-table) def
    (declare (ignorable _))
    (mapcar #'kerdef-name (remove-if-not #'kerdef-symbol-macro-p var-table))))

(defun kernel-definition-symbol-macro-expansion (name def)
  (unless (kernel-definition-symbol-macro-exists-p name def)
    (error "undefined kernel definition symbol macro: ~A" name))
  (kerdef-symbol-macro-expansion (lookup-kernel-definition :symbol-macro name def)))


;;;
;;; Compiling
;;;

(defun compile-function-specifier (return-type)
  (unless (valid-type-p return-type)
    (error (format nil "invalid return type: ~A" return-type)))
  (if (eq return-type 'void)
      "__global__"
      "__device__"))

(defun compile-type (type)
  (unless (valid-type-p type)
    (error (format nil "invalid type: ~A" type)))
  (cond ((eq type 'curand-state-xorwow)
         "curandStateXORWOW_t")
        ((eq type 'curand-state-xorwow*)
         "curandStateXORWOW_t *")
        (t
         (compile-identifier (princ-to-string type)))))

(defun compile-argument (arg)
  (destructuring-bind (var type) arg
    (format nil "~A ~A" (compile-type type) (compile-identifier var))))

(defun compile-arguments (args)
  (join ", " (mapcar #'compile-argument args)))

(defun compile-function-declaration (name def)
  (let ((c-name (kernel-definition-function-c-name name def))
        (arguments (kernel-definition-function-arguments name def))
        (return-type (kernel-definition-function-return-type name def)))
    (let ((specifier (compile-function-specifier return-type))
          (type (compile-type return-type))
          (args (compile-arguments arguments)))
      (format nil "~A ~A ~A (~A)" specifier type c-name args))))

(defun compile-kernel-constant (name def)
  (let ((type (kernel-definition-constant-type name def))
        (exp (kernel-definition-constant-expression name def)))
    (let ((var-env  (make-variable-environment-with-kernel-definition nil def))
          (func-env (make-function-environment-with-kernel-definition def)))
      (let ((type2 (compile-type type))
            (name2 (compile-identifier name))
            (exp2  (compile-expression exp var-env func-env)))
        (format nil "static const ~A ~A = ~A;" type2 name2 exp2)))))

(defun compile-kernel-constants (def)
  (mapcar #'(lambda (name)
              (compile-kernel-constant name def))
          (reverse (kernel-definition-constant-names def))))

(defun compile-kernel-function-prototype (name def)
  (format nil "extern \"C\" ~A;"
          (compile-function-declaration name def)))

(defun compile-kernel-function-prototypes (def)
  (mapcar #'(lambda (name)
              (compile-kernel-function-prototype name def))
          (reverse (kernel-definition-function-names def))))

(defun compile-function-statements (name def)
  (let ((var-env  (make-variable-environment-with-kernel-definition name def))
        (func-env (make-function-environment-with-kernel-definition def)))
    (mapcar #'(lambda (stmt)
                (compile-statement stmt var-env func-env))
            (kernel-definition-function-body name def))))

(defun compile-kernel-function (name def)
  (let ((declaration (compile-function-declaration name def))
        (statements  (mapcar #'(lambda (stmt)
                                 (indent 2 stmt))
                             (compile-function-statements name def))))
    (unlines `(,declaration
               "{"
               ,@statements
               "}"
               ""))))

(defun compile-kernel-functions (def)
  (mapcar #'(lambda (name)
              (compile-kernel-function name def))
          (reverse (kernel-definition-function-names def))))

(defun compile-kernel-definition (def)
  (unlines `("#include \"int.h\""
             "#include \"float.h\""
             "#include \"float3.h\""
             "#include \"float4.h\""
             "#include \"double.h\""
             "#include \"double3.h\""
             "#include \"double4.h\""
             "#include \"curand.h\""
             ""
             ,@(compile-kernel-function-prototypes def)
             ""
             ,@(compile-kernel-functions def))))
  

;;; compile statement

(defun compile-statement (stmt var-env func-env)
  (cond
    ((macro-form-p stmt func-env) (compile-macro stmt var-env func-env :statement-p t))
    ((if-p stmt) (compile-if stmt var-env func-env))
    ((let-p stmt) (compile-let stmt var-env func-env))
    ((let*-p stmt) (compile-let* stmt var-env func-env))
    ((symbol-macrolet-p stmt) (compile-symbol-macrolet stmt var-env func-env))
    ((do-p stmt) (compile-do stmt var-env func-env))
    ((with-shared-memory-p stmt) (compile-with-shared-memory stmt var-env func-env))
    ((set-p stmt) (compile-set stmt var-env func-env))
    ((progn-p stmt) (compile-progn stmt var-env func-env))
    ((return-p stmt) (compile-return stmt var-env func-env))
    ((syncthreads-p stmt) (compile-syncthreads stmt))
    ((function-p stmt) (compile-function stmt var-env func-env :statement-p t))
    (t (error "invalid statement: ~A" stmt))))


;;; if statement

(defun if-p (stmt)
  (cl-pattern:match stmt
    (('if _ _) t)
    (('if _ _ _) t)
    (_ nil)))

(defun if-test-expression (stmt)
  (cl-pattern:match stmt
    (('if test-exp _) test-exp)
    (('if test-exp _ _) test-exp)
    (_ (error "invalid statement: ~A" stmt))))

(defun if-then-statement (stmt)
  (cl-pattern:match stmt
    (('if _ then-stmt) then-stmt)
    (('if _ then-stmt _) then-stmt)
    (_ (error "invalid statement: ~A" stmt))))

(defun if-else-statement (stmt)
  (cl-pattern:match stmt
    (('if _ _) nil)
    (('if _ _ else-stmt) else-stmt)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-if (stmt var-env func-env)
  (let ((test-exp  (if-test-expression stmt))
        (then-stmt (if-then-statement stmt))
        (else-stmt (if-else-statement stmt)))
    (let ((test-type (type-of-expression test-exp var-env func-env)))
      (unless (eq test-type 'bool)
        (error "invalid type: type of test-form is ~A, not ~A" test-type 'bool)))
    (unlines (format nil "if (~A) {"
                     (compile-expression test-exp var-env func-env))
             (indent 2 (compile-statement then-stmt var-env func-env))
             (and else-stmt "} else {")
             (and else-stmt
                  (indent 2 (compile-statement else-stmt var-env func-env)))
             "}")))


;;; let statement

(defun let-p (stmt)
  (cl-pattern:match stmt
    (('let . _) t)
    (_ nil)))

(defun let*-p (stmt)
  (cl-pattern:match stmt
    (('let* . _) t)
    (_ nil)))

(defun let-bindings (stmt)
  (cl-pattern:match stmt
    (('let bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun let*-bindings (stmt)
  (cl-pattern:match stmt
    (('let* bindings . _) bindings)))

(defun let-statements (stmt)
  (cl-pattern:match stmt
    (('let _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun let*-statements (stmt)
  (cl-pattern:match stmt
    (('let* _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun %compile-assignment (var exp type var-env func-env)
  (let ((var2  (compile-identifier var))
        (exp2  (compile-expression exp var-env func-env))
        (type2 (compile-type type)))
    (format nil "~A ~A = ~A;" type2 var2 exp2)))

(defun compile-let-assignments (bindings var-env func-env)
  (labels ((aux (binding)
             (cl-pattern:match binding
               ((var exp) (let ((type (type-of-expression exp var-env func-env)))
                            (%compile-assignment var exp type var-env func-env)))
               (_ (error "invalid let binding: ~A" binding)))))
    (let ((compiled-assignments (mapcar #'aux bindings)))
      (apply #'unlines compiled-assignments))))

(defun compile-let-statements (stmts var-env func-env)
  (compile-statement `(progn ,@stmts) var-env func-env))

(defun compile-let (stmt var-env func-env)
  (labels ((aux (binding)
             (cl-pattern:match binding
               ((var exp) (let ((type (type-of-expression exp var-env func-env)))
                            (list var :variable type)))
               (_ (error "invalid let binding: ~A" binding)))))
    (let ((bindings  (let-bindings stmt))
          (let-stmts (let-statements stmt)))
      (let ((var-env2 (bulk-add-variable-environment (mapcar #'aux bindings) var-env)))
        (let ((assignments (compile-let-assignments bindings var-env func-env))
              (compiled-stmts (compile-let-statements let-stmts var-env2 func-env)))
          (unlines "{"
                   (indent 2 assignments)
                   (indent 2 compiled-stmts)
                   "}"))))))

(defun compile-let*-binding (bindings stmts var-env func-env)
  (cl-pattern:match bindings
    (((var exp) . rest)
     (let ((type (type-of-expression exp var-env func-env)))
       (let ((assignment (%compile-assignment var exp type var-env func-env))
             (var-env2   (add-variable-to-variable-environment var type var-env)))
         (unlines assignment
                  (%compile-let* rest stmts var-env2 func-env)))))
    (_ (error "invalid bindings: ~A" bindings))))

(defun %compile-let* (bindings stmts var-env func-env)
  (if bindings
      (compile-let*-binding bindings stmts var-env func-env)
      (compile-let-statements stmts var-env func-env)))

(defun compile-let* (stmt var-env func-env)
  (let ((bindings  (let*-bindings stmt))
        (let-stmts (let*-statements stmt)))
    (unlines "{"
             (indent 2 (%compile-let* bindings let-stmts var-env func-env))
             "}")))


;;; symbol-macrolet statement

(defun symbol-macrolet-p (stmt)
  (cl-pattern:match stmt
    (('symbol-macrolet . _) t)
    (_ nil)))

(defun symbol-macrolet-bindings (stmt)
  (cl-pattern:match stmt
    (('symbol-macrolet bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun symbol-macrolet-statements (stmt)
  (cl-pattern:match stmt
    (('symbol-macrolet _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-symbol-macrolet (stmt var-env func-env)
  (labels ((aux (binding)
             (cl-pattern:match binding
               ((name expansion) (list name :symbol-macro expansion))
               (_ (error "invalid symbol-macrolet binding: ~A" binding)))))
    (let ((bindings (symbol-macrolet-bindings stmt))
          (stmts    (symbol-macrolet-statements stmt)))
      (let ((var-env2 (bulk-add-variable-environment (mapcar #'aux bindings) var-env)))
        (compile-statement `(progn ,@stmts) var-env2 func-env)))))


;;; set statement

(defun set-p (stmt)
  (cl-pattern:match stmt
    (('set _ _) t)
    (_ nil)))

(defun set-place (stmt)
  (cl-pattern:match stmt
    (('set place _) place)
    (_ (error "invalid statement: ~A" stmt))))

(defun set-expression (stmt)
  (cl-pattern:match stmt
    (('set _ exp) exp)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-set (stmt var-env func-env)
  (let ((place (set-place stmt))
        (exp (set-expression stmt)))
    (let ((place-type (type-of-expression place var-env func-env))
          (exp-type   (type-of-expression exp   var-env func-env)))
      (unless (eq place-type exp-type)
        (error "invalid types: type of the place is ~A but that of the expression is ~A" place-type exp-type)))
    (format nil "~A = ~A;" (compile-place place var-env func-env)
                           (compile-expression exp var-env func-env))))

(defun compile-place (place var-env func-env)
  (cond ((symbol-place-p place) (compile-symbol-place place var-env func-env))
        ((vector-place-p place) (compile-vector-place place var-env func-env))
        ((array-place-p place)  (compile-array-place place var-env func-env))
        (t (error "invalid place: ~A" place))))

(defun symbol-place-p (place)
  (symbol-p place))

(defun vector-place-p (place)
  (vector-variable-reference-p place))

(defun array-place-p (place)
  (array-variable-reference-p place))

(defun compile-symbol-place (place var-env func-env)
  (compile-symbol place var-env func-env))

(defun compile-vector-place (place var-env func-env)
  (compile-vector-variable-reference place var-env func-env))

(defun compile-array-place (place var-env func-env)
  (compile-array-variable-reference place var-env func-env))


;;; progn statement

(defun progn-p (stmt)
  (cl-pattern:match stmt
    (('progn . _) t)
    (_ nil)))

(defun progn-statements (stmt)
  (cl-pattern:match stmt
    (('progn . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-progn-statements (stmts var-env func-env)
  (let ((compiled-stmts (mapcar #'(lambda (stmt)
                                    (compile-statement stmt var-env func-env))
                                stmts)))
    (unlines compiled-stmts stmts)))

(defun compile-progn (stmt var-env func-env)
  (let ((stmts (progn-statements stmt)))
    (compile-progn-statements stmts var-env func-env)))


;;; return statement

(defun return-p (stmt)
  (cl-pattern:match stmt
    (('return) t)
    (('return _) t)
    (_ nil)))

(defun compile-return (stmt var-env func-env)
  (cl-pattern:match stmt
    (('return) "return;")
    (('return exp) (format nil "return ~A;"
                               (compile-expression exp var-env func-env)))
    (_ (error "invalid statement: ~A" stmt))))


;;; do statement

(defun do-p (stmt)
  (cl-pattern:match stmt
    (('do . _) t)
    (_ nil)))

(defun do-bindings (stmt)
  (cl-pattern:match stmt
    (('do bindings . _) bindings)
    (_ (error "invalid statement: ~A" stmt))))

(defun do-var-types (stmt var-env func-env)
  (labels ((do-var-type (binding)
             (list (do-binding-var binding)
                   :variable
                   (do-binding-type binding var-env func-env))))
    (mapcar #'do-var-type (do-bindings stmt))))

(defun do-binding-var (binding)
  (cl-pattern:match binding
    ((var _)   var)
    ((var _ _) var)
    (_ (error "invalid binding: ~A" binding))))

(defun do-binding-type (binding var-env func-env)
  (type-of-expression (do-binding-init-form binding) var-env func-env))

(defun do-binding-init-form (binding)
  (cl-pattern:match binding
    ((_ init-form)   init-form)
    ((_ init-form _) init-form)
    (_ (error "invalid binding: ~A" binding))))

(defun do-binding-step-form (binding)
  (cl-pattern:match binding
    ((_ _)           nil)
    ((_ _ step-form) step-form)
    (_ (error "invalid binding: ~A" binding))))

(defun do-test-form (stmt)
  (cl-pattern:match stmt
    (('do _ (test-form) . _) test-form)
    (_ (error "invalid statement: ~A" stmt))))

(defun do-statements (stmt)
  (cl-pattern:match stmt
    (('do _ _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-do (stmt var-env func-env)
  (let ((var-env2 (bulk-add-variable-environment (do-var-types stmt var-env func-env) var-env)))
    (let ((init-part (compile-do-init-part stmt var-env func-env))
          (test-part (compile-do-test-part stmt var-env2 func-env))
          (step-part (compile-do-step-part stmt var-env2 func-env)))
      (unlines (format nil "for ( ~A; ~A; ~A )" init-part test-part step-part)
               "{"
               (indent 2 (compile-do-statements stmt var-env2 func-env))
               "}"))))

(defun compile-do-init-part (stmt var-env func-env)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (type (do-binding-type binding var-env func-env))
                   (init-form (do-binding-init-form binding)))
               (format nil "~A ~A = ~A" (compile-type type)
                                        (compile-identifier var)
                                        (compile-expression init-form var-env func-env)))))
    (join ", " (mapcar #'aux (do-bindings stmt)))))

(defun compile-do-test-part (stmt var-env func-env)
  (let ((test-form (do-test-form stmt)))
    (format nil "! ~A" (compile-expression test-form var-env func-env))))

(defun compile-do-step-part (stmt var-env func-env)
  (labels ((aux (binding)
             (let ((var (do-binding-var binding))
                   (step-form (do-binding-step-form binding)))
               (format nil "~A = ~A" (compile-identifier var)
                                     (compile-expression step-form var-env func-env)))))
    (join ", " (mapcar #'aux (remove-if-not #'do-binding-step-form (do-bindings stmt))))))

(defun compile-do-statements (stmt var-env func-env)
  (compile-progn-statements (do-statements stmt) var-env func-env))


;;; with-shared-memory statement

(defun with-shared-memory-p (stmt)
  (cl-pattern:match stmt
    (('with-shared-memory . _) t)
    (_ nil)))

(defun with-shared-memory-specs (stmt)
  (cl-pattern:match stmt
    (('with-shared-memory specs . _) specs)
    (_ (error "invalid statement: ~A" stmt))))

(defun with-shared-memory-statements (stmt)
  (cl-pattern:match stmt
    (('with-shared-memory _ . stmts) stmts)
    (_ (error "invalid statement: ~A" stmt))))

(defun compile-with-shared-memory-statements (stmts var-env func-env)
  (compile-let-statements stmts var-env func-env))

(defun compile-with-shared-memory-spec (specs stmts var-env func-env)
  (cl-pattern:match specs
    (((var type . sizes) . rest)
     (let* ((type2 (add-star type (length sizes)))
            (var-env2 (add-variable-to-variable-environment var type2 var-env)))
       (unlines (format nil "__shared__ ~A ~A~{[~A]~};"
                            (compile-type type)
                            (compile-identifier var)
                            (mapcar #'(lambda (exp)
                                        (compile-expression exp var-env func-env))
                                    sizes))
                (%compile-with-shared-memory rest stmts var-env2 func-env))))
    (_ (error "invalid shared memory specs: ~A" specs))))

(defun %compile-with-shared-memory (specs stmts var-env func-env)
  (if (null specs)
      (compile-with-shared-memory-statements stmts var-env func-env)
      (compile-with-shared-memory-spec specs stmts var-env func-env)))

(defun compile-with-shared-memory (stmt var-env func-env)
  (let ((specs (with-shared-memory-specs stmt))
        (stmts (with-shared-memory-statements stmt)))
    (unlines "{"
             (indent 2 (%compile-with-shared-memory specs stmts var-env func-env))
             "}")))


;;; compile syncthreads

(defun syncthreads-p (stmt)
  (cl-pattern:match stmt
    (('syncthreads) t)
    (_ nil)))

(defun compile-syncthreads (stmt)
  (declare (ignorable stmt))
  "__syncthreads();")


;;; compile function

(defun function-p (form)
  (and (listp form)
       (car form)
       (symbolp (car form))))

(defun defined-function-p (form func-env)
  (or (built-in-function-p form)
      (user-function-p form func-env)))

(defun built-in-function-p (form)
  (cl-pattern:match form
    ((op . _) (and (getf +built-in-functions+ op) t))
    (_ nil)))

(defun user-function-p (form func-env)
  (cl-pattern:match form
    ((op . _) (function-environment-function-exists-p op func-env))
    (_ nil)))

(defun function-operator (form)
  (unless (function-p form)
    (error "invalid statement or expression: ~A" form))
  (car form))

(defun function-operands (form)
  (unless (function-p form)
    (error "invalid statement or expression: ~A" form))
  (cdr form))

(defun compile-function (form var-env func-env &key (statement-p nil))
  (unless (defined-function-p form func-env)
    (error "undefined function: ~A" form))
  (let ((code (if (built-in-function-p form)
                  (compile-built-in-function form var-env func-env)
                  (compile-user-function form var-env func-env))))
    (if statement-p
        (format nil "~A;" code)
        code)))

(defun compile-built-in-function (form var-env func-env)
  (let ((op (function-operator form)))
    (cond
      ((built-in-function-infix-p form var-env func-env)
       (compile-built-in-infix-function form var-env func-env))
      ((built-in-function-prefix-p form var-env func-env)
       (compile-built-in-prefix-function form var-env func-env))
      (t (error "invalid built-in function: ~A" op)))))

(defun compile-built-in-infix-function (form var-env func-env)
  (let ((operands (function-operands form)))
    (let ((op  (built-in-function-c-string form var-env func-env))
          (lhe (compile-expression (car operands) var-env func-env))
          (rhe (compile-expression (cadr operands) var-env func-env)))
      (format nil "(~A ~A ~A)" lhe op rhe))))

(defun compile-built-in-prefix-function (form var-env func-env)
  (let ((operands (function-operands form)))
    (format nil "~A (~A)"
            (built-in-function-c-string form var-env func-env)
            (compile-operands operands var-env func-env))))

(defun type-of-operands (operands var-env func-env)
  (mapcar #'(lambda (exp)
              (type-of-expression exp var-env func-env))
          operands))

(defun compile-operands (operands var-env func-env)
  (join ", " (mapcar #'(lambda (exp)
                         (compile-expression exp var-env func-env))
                     operands)))

(defun compile-user-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((expected-types (function-environment-function-argument-types operator func-env))
          (actual-types (type-of-operands operands var-env func-env)))
      (unless (equal expected-types actual-types)
        (error "invalid arguments: ~A" form)))
    (let ((func (function-environment-function-c-name operator func-env))
          (compiled-operands (compile-operands operands var-env func-env)))
      (format nil "~A (~A)" func compiled-operands))))


;;; compile macro

(defun macro-form-p (form func-env)
  "Returns t if the given form is a macro form. The macro used in the
form may be an user-defined macro under the given kernel definition or
a built-in macro."
  (or (built-in-macro-p form)
      (user-macro-p form func-env)))

(defun built-in-macro-p (form)
  (cl-pattern:match form
    ((op . _) (and (getf +built-in-macros+ op) t))
    (_ nil)))

(defun user-macro-p (form func-env)
  (cl-pattern:match form
    ((op . _) (function-environment-macro-exists-p op func-env))
    (_ nil)))

(defun macro-operator (form func-env)
  (unless (macro-form-p form func-env)
    (error "undefined macro form: ~A" form))
  (car form))

(defun macro-operands (form func-env)
  (unless (macro-form-p form func-env)
    (error "undefined macro form: ~A" form))
  (cdr form))

(defun compile-macro (form var-env func-env &key (statement-p nil))
  (unless (macro-form-p form func-env)
    (error "undefined macro: ~A" form))
  (if statement-p
      (compile-statement  (%expand-macro-1 form func-env) var-env func-env)
      (compile-expression (%expand-macro-1 form func-env) var-env func-env)))

(defun %expand-built-in-macro-1 (form func-env)
  (let ((operator (macro-operator form func-env))
        (operands (macro-operands form func-env)))
    (let ((expander (built-in-macro-expander operator)))
      (values (funcall expander operands) t))))

(defun %expand-user-macro-1 (form func-env)
  (let ((operator (macro-operator form func-env))
        (operands (macro-operands form func-env)))
    (let ((expander (function-environment-macro-expander operator func-env)))
      (values (funcall expander operands) t))))

(defun %expand-macro-1 (form func-env)
  (if (macro-form-p form func-env)
      (if (built-in-macro-p form)
          (%expand-built-in-macro-1 form func-env)
          (%expand-user-macro-1 form func-env))
      (values form nil)))

(defun expand-macro-1 (form def)
  "If a form is a macro form, then EXPAND-MACRO-1 expands the macro
form call once, and returns the macro expansion and true as values.
Otherwise, returns the given form and false as values."
  (let ((func-env (make-function-environment-with-kernel-definition def)))
    (%expand-macro-1 form func-env)))

(defun %expand-macro (form func-env)
  (if (macro-form-p form func-env)
      (values (%expand-macro (%expand-macro-1 form func-env) func-env) t)
      (values form nil)))

(defun expand-macro (form def)
  "If a form is a macro form, then EXPAND-MACRO repeatedly expands
the macro form until it is no longer a macro form, and returns the
macro expansion and true as values. Otherwise, returns the given form
and false as values."
  (let ((func-env (make-function-environment-with-kernel-definition def)))
    (%expand-macro form func-env)))


;;; built-in functions
;;;   <built-in-functions>  ::= plist { <function-name> => <function-info> }
;;;   <function-info>       ::= (<infix-p> <function-candidates>)
;;;   <function-candidates> ::= (<function-candidate>*)
;;;   <function-candidate>  ::= (<arg-types> <return-type> <function-c-name>)
;;;   <arg-types>           ::= (<arg-type>*)

(defparameter +built-in-functions+
  '(%add (((int    int)    int    t   "+")
          ((float  float)  float  t   "+")
          ((float3 float3) float3 nil "float3_add")
          ((float4 float4) float4 nil "float4_add")
          ((double  double)  double  t   "+")
          ((double3 double3) double3 nil "double3_add")
          ((double4 double4) double4 nil "double4_add"))
    %sub (((int    int)    int    t   "-")
          ((float  float)  float  t   "-")
          ((float3 float3) float3 nil "float3_sub")
          ((float4 float4) float4 nil "float4_sub")
          ((double  double)  double  t   "-")
          ((double3 double3) double3 nil "double3_sub")
          ((double4 double4) double4 nil "double4_sub"))
    %mul (((int    int)    int    t   "*")
          ((float  float)  float  t   "*")
          ((float3 float)  float3 nil "float3_scale")
          ((float  float3) float3 nil "float3_scale_flipped")
          ((float4 float)  float4 nil "float4_scale")
          ((float  float4) float4 nil "float4_scale_flipped")
          ((double  double)  double  t   "*")
          ((double3 double)  double3 nil "double3_scale")
          ((double  double3) double3 nil "double3_scale_flipped")
          ((double4 double)  double4 nil "double4_scale")
          ((double  double4) double4 nil "double4_scale_flipped"))
    %div (((int    int)    int    t   "/")
          ((float  float)  float  t   "/")
          ((float3 float)  float3 nil "float3_scale_inverted")
          ((float4 float)  float4 nil "float4_scale_inverted")
          ((double  double)  double  t   "/")
          ((double3 double)  double3 nil "double3_scale_inverted")
          ((double4 double)  double4 nil "double4_scale_inverted"))
    %negate (((int)    int    nil "int_negate")
             ((float)  float  nil "float_negate")
             ((float3) float3 nil "float3_negate")
             ((float4) float4 nil "float4_negate")
             ((double)  double  nil "double_negate")
             ((double3) double3 nil "double3_negate")
             ((double4) double4 nil "double4_negate"))
    %recip (((int)    int    nil "int_recip")
            ((float)  float  nil "float_recip")
            ((float3) float3 nil "float3_recip")
            ((float4) float4 nil "float4_recip")
            ((double)  double  nil "double_recip")
            ((double3) double3 nil "double3_recip")
            ((double4) double4 nil "double4_recip"))
    =    (((int   int)   bool t "==")
          ((float float) bool t "==")
          ((double double) bool t "=="))
    /=   (((int   int)   bool t "!=")
          ((float float) bool t "!=")
          ((double double) bool t "!="))
    <    (((int   int)   bool t "<")
          ((float float) bool t "<")
          ((double double) bool t "<"))
    >    (((int   int)   bool t ">")
          ((float float) bool t ">")
          ((double double) bool t ">"))
    <=   (((int   int)   bool t "<=")
          ((float float) bool t "<=")
          ((double double) bool t "<="))
    >=   (((int   int)   bool t ">=")
          ((float float) bool t ">=")
          ((double double) bool t ">="))
    not  (((bool) bool nil "!"))
    exp  (((float) float nil "expf")
          ((double) double nil "exp"))
    log  (((float) float nil "logf")
          ((double) double nil "log"))
    expt   (((float float) float nil "powf")
            ((double double) double nil "pow"))
    rsqrtf (((float) float nil "rsqrtf")
            ((double) double nil "rsqrt"))
    sqrt   (((float) float nil "sqrtf")
            ((double) double nil "sqrt"))
    floor  (((float) int   nil "floorf")
            ((double) int   nil "floor"))
    atomic-add (((int* int) int nil "atomicAdd"))
    pointer (((int)   int*   nil "&")
             ((float) float* nil "&")
             ((double) double* nil "&")
             ((curand-state-xorwow) curand-state-xorwow* nil "&"))
    float3 (((float float float) float3 nil "make_float3"))
    float4 (((float float float float) float4 nil "make_float4"))
    double3 (((double double double) double3 nil "make_double3"))
    double4 (((double double double double) double4 nil "make_double4"))
    double-to-int-rn (((double) int nil "__double2int_rn"))
    dot (((float3 float3) float nil "float3_dot")
         ((float4 float4) float nil "float4_dot")
         ((double3 double3) double nil "double3_dot")
         ((double4 double4) double nil "double4_dot"))
    ;; It's :UNSIGNED-LONG-LONG, but this wrapper function only
    ;; supports INT.
    curand-init-xorwow (((int int int curand-state-xorwow*) void nil
                         "curand_init_xorwow"))
    curand-uniform-float-xorwow (((curand-state-xorwow*) float nil
                                  "curand_uniform_float_xorwow"))
    curand-uniform-double-xorwow (((curand-state-xorwow*) double nil
                                   "curand_uniform_double_xorwow"))))

(defun function-candidates (op)
  (or (getf +built-in-functions+ op)
      (error "invalid function: ~A" op)))

(defun inferred-function (form var-env func-env)
  (let ((operator (function-operator form))
        (operands (function-operands form)))
    (let ((candidates (function-candidates operator))
          (types (type-of-operands operands var-env func-env)))
      (or (find types candidates :key #'car :test #'equal)
          (error "invalid function application: ~A" form)))))

(defun inferred-function-argument-types (fun)
  (car fun))

(defun inferred-function-return-type (fun)
  (cadr fun))

(defun inferred-function-infix-p (fun)
  (caddr fun))

(defun inferred-function-prefix-p (fun)
  (not (inferred-function-infix-p fun)))

(defun inferred-function-c-string (fun)
  (cadddr fun))

(defun built-in-function-argument-types (form var-env func-env)
  (inferred-function-argument-types (inferred-function form var-env func-env)))

(defun built-in-function-return-type (form var-env func-env)
  (inferred-function-return-type (inferred-function form var-env func-env)))

(defun built-in-function-infix-p (form var-env func-env)
  (inferred-function-infix-p (inferred-function form var-env func-env)))

(defun built-in-function-prefix-p (form var-env func-env)
  (inferred-function-prefix-p (inferred-function form var-env func-env)))

(defun built-in-function-c-string (form var-env func-env)
  (inferred-function-c-string (inferred-function form var-env func-env)))


;;; built-in macros
;;;   <built-in-macros> ::= plist { <macro-name> => <macro-expander> }

(defvar +built-in-macros+
  (list '+ (lambda (args)
             (cl-pattern:match args
               (() 0)
               ((a1) a1)
               ((a1 a2) `(%add ,a1 ,a2))
               ((a1 a2 . rest) `(+ (%add ,a1 ,a2) ,@rest))))
        '- (lambda (args)
             (cl-pattern:match args
               (() (error "invalid number of arguments: 0"))
               ((a1) `(%negate ,a1))
               ((a1 a2) `(%sub ,a1 ,a2))
               ((a1 a2 . rest) `(- (%sub ,a1 ,a2) ,@rest))))
        '* (lambda (args)
             (cl-pattern:match args
               (() 1)
               ((a1) a1)
               ((a1 a2) `(%mul ,a1 ,a2))
               ((a1 a2 . rest) `(* (%mul ,a1 ,a2) ,@rest))))
        '/ (lambda (args)
             (cl-pattern:match args
               (() (error "invalid number of arguments: 0"))
               ((a1) `(%recip ,a1))
               ((a1 a2) `(%div ,a1 ,a2))
               ((a1 a2 . rest) `(/ (%div ,a1 ,a2) ,@rest))))))

(defun built-in-macro-expander (name)
  (or (getf +built-in-macros+ name)
      (error "invalid macro name: ~A" name)))


;;; compile expression

(defun compile-expression (exp var-env func-env)
  (cond
    ((macro-form-p exp func-env) (compile-macro exp var-env func-env))
    ((literal-p exp) (compile-literal exp))
    ((cuda-dimension-p exp) (compile-cuda-dimension exp))
    ((symbol-p exp) (compile-symbol exp var-env func-env))
    ((variable-reference-p exp)
     (compile-variable-reference exp var-env func-env))
    ((inline-if-p exp) (compile-inline-if exp var-env func-env))
    ((function-p exp) (compile-function exp var-env func-env))
    (t (error "invalid expression: ~A" exp))))

(defun variable-p (exp var-env)
  (variable-environment-variable-exists-p exp var-env))

(defun compile-variable (exp var-env)
  (unless (variable-environment-variable-exists-p exp var-env)
    (error "undefined variable: ~A" exp))
  (compile-identifier exp))

(defun constant-p (exp var-env)
  (variable-environment-constant-exists-p exp var-env))

(defun compile-constant (exp var-env)
  (unless (variable-environment-constant-exists-p exp var-env)
    (error "undefined constant: ~A" exp))
  (compile-identifier exp))

(defun symbol-macro-p (exp var-env)
  (variable-environment-symbol-macro-exists-p exp var-env))

(defun compile-symbol-macro (exp var-env func-env)
  (unless (variable-environment-symbol-macro-exists-p exp var-env)
    (error "undefined symbol macro: ~A" exp))
  (let ((expansion (variable-environment-symbol-macro-expansion exp var-env)))
    (compile-expression expansion var-env func-env)))

(defun symbol-p (exp)
  (symbolp exp))

(defun compile-symbol (exp var-env func-env)
  (cond
    ((variable-p exp var-env) (compile-variable exp var-env))
    ((constant-p exp var-env) (compile-constant exp var-env))
    ((symbol-macro-p exp var-env) (compile-symbol-macro exp var-env func-env))
    (t (error "undefined variable: ~A" exp))))

(defun literal-p (exp)
  (or (bool-literal-p exp)
      (int-literal-p exp)
      (float-literal-p exp)
      (double-literal-p exp)))

(defun bool-literal-p (exp)
  (typep exp 'boolean))

(defun int-literal-p (exp)
  (typep exp 'fixnum))

(defun float-literal-p (exp)
  (typep exp 'single-float))

(defun double-literal-p (exp)
  (typep exp 'double-float))

(defun compile-bool-literal (exp)
  (unless (typep exp 'boolean)
    (error "invalid literal: ~A" exp))
  (if exp "true" "false"))

(defun compile-int-literal (exp)
  (princ-to-string exp))

(defun compile-float-literal (exp)
  (princ-to-string exp))

(defun compile-double-literal (exp)
  (format nil "(double)~S" (float exp 0.0)))

(defun compile-literal (exp)
  (cond ((bool-literal-p  exp) (compile-bool-literal exp))
        ((int-literal-p   exp) (compile-int-literal exp))
        ((float-literal-p exp) (compile-float-literal exp))
        ((double-literal-p exp) (compile-double-literal exp))
        (t (error "invalid literal: ~A" exp))))

(defun cuda-dimension-p (exp)
  (or (grid-dim-p exp) (block-dim-p exp) (block-idx-p exp) (thread-idx-p exp)))

(defun grid-dim-p (exp)
  (find exp '(grid-dim-x grid-dim-y grid-dim-z)))

(defun block-dim-p (exp)
  (find exp '(block-dim-x block-dim-y block-dim-z)))

(defun block-idx-p (exp)
  (find exp '(block-idx-x block-idx-y block-idx-z)))

(defun thread-idx-p (exp)
  (find exp '(thread-idx-x thread-idx-y thread-idx-z)))

(defun compile-cuda-dimension (exp)
  (case exp
    (grid-dim-x "gridDim.x")
    (grid-dim-y "gridDim.y")
    (grid-dim-z "gridDim.z")
    (block-dim-x "blockDim.x")
    (block-dim-y "blockDim.y")
    (block-dim-z "blockDim.z")
    (block-idx-x "blockIdx.x")
    (block-idx-y "blockIdx.y")
    (block-idx-z "blockIdx.z")
    (thread-idx-x "threadIdx.x")
    (thread-idx-y "threadIdx.y")
    (thread-idx-z "threadIdx.z")
    (t (error "invalid expression: ~A" exp))))

(defun variable-reference-p (exp)
  (or (vector-variable-reference-p exp)
      (array-variable-reference-p exp)))

(defun vector-variable-reference-p (exp)
  (cl-pattern:match exp
    ((selector _) (valid-vector-type-selector-p selector))
    (_ nil)))

(defun array-variable-reference-p (exp) 
  (cl-pattern:match exp
    (('aref . _) t)
    (_ nil)))

(defun compile-variable-reference (exp var-env func-env)
  (cond ((vector-variable-reference-p exp)
         (compile-vector-variable-reference exp var-env func-env))
        ((array-variable-reference-p exp)
         (compile-array-variable-reference exp var-env func-env))
        (t (error "invalid expression: ~A" exp))))

(defun compile-vector-selector (selector)
  (unless (valid-vector-type-selector-p selector)
    (error "invalid selector: ~A" selector))
  (string-downcase (subseq (reverse (princ-to-string selector)) 0 1)))

(defun compile-vector-variable-reference (form var-env func-env)
  (cl-pattern:match form
    ((selector exp)
     (let ((selector-type (vector-type-selector-type selector))
           (exp-type      (type-of-expression exp var-env func-env)))
       (unless (eq selector-type exp-type)
         (error "invalid variable reference: ~A" form))
       (format nil "~A.~A" (compile-expression exp var-env func-env)
                           (compile-vector-selector selector))))
    (_ (error "invalid variable reference: ~A" form))))

(defun compile-array-variable-reference (form var-env func-env)
  (cl-pattern:match form
    (('aref _)
     (error "invalid variable reference: ~A" form))
    (('aref exp . idxs)
     (let ((type (type-of-expression exp var-env func-env)))
       (unless (= (array-type-dimension type) (length idxs))
         (error "invalid dimension: ~A" form))
       (format nil "~A~{[~A]~}"
                   (compile-expression exp var-env func-env)
                   (mapcar #'(lambda (idx)
                               (compile-expression idx var-env func-env)) idxs))))
    (_ (error "invalid variable reference: ~A" form))))

(defun inline-if-p (exp)
  (cl-pattern:match exp
    (('if _ _ _) t)
    (_ nil)))

(defun inline-if-test-expression (exp)
  (cl-pattern:match exp
    (('if test-exp _ _) test-exp)
    (_ (error "invalid expression: ~A" exp))))

(defun inline-if-then-expression (exp)
  (cl-pattern:match exp
    (('if _ then-exp _) then-exp)
    (_ (error "invalid expression: ~A" exp))))

(defun inline-if-else-expression (exp)
  (cl-pattern:match exp
    (('if _ _ else-exp) else-exp)
    (_ (error "invalid expression: ~A" exp))))

(defun compile-inline-if (exp var-env func-env)
  (let ((test-exp (inline-if-test-expression exp))
        (then-exp (inline-if-then-expression exp))
        (else-exp (inline-if-else-expression exp)))
    (let ((test-type (type-of-expression test-exp var-env func-env))
          (then-type (type-of-expression then-exp var-env func-env))
          (else-type (type-of-expression else-exp var-env func-env)))
      (unless (eq test-type 'bool)
        (error "invalid type: type of test-form is ~A, not ~A" test-type 'bool))
      (unless (eq then-type else-type)
        (error "invalid types: type of then-form is ~A but that of else-form is ~A" then-type else-type)))
    (format nil "(~A ? ~A : ~A)"
            (compile-expression test-exp var-env func-env)
            (compile-expression then-exp var-env func-env)
            (compile-expression else-exp var-env func-env))))


;;;
;;; Type of expression
;;;

(defun type-of-expression (exp var-env func-env)
  (cond ((macro-form-p exp func-env) (type-of-macro-form exp var-env func-env))
        ((literal-p exp) (type-of-literal exp))
        ((cuda-dimension-p exp) 'int)
        ((symbol-p exp) (type-of-symbol exp var-env func-env))
        ((variable-reference-p exp) (type-of-variable-reference exp var-env func-env))
        ((inline-if-p exp) (type-of-inline-if exp var-env func-env))
        ((function-p exp) (type-of-function exp var-env func-env))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-literal (exp)
  (cond ((bool-literal-p exp) 'bool)
        ((int-literal-p exp) 'int)
        ((float-literal-p exp) 'float)
        ((double-literal-p exp) 'double)
        (t (error "invalid expression: ~A" exp))))

(defun type-of-variable (exp var-env)
  (unless (variable-environment-variable-exists-p exp var-env)
    (error "undefined variable: ~A" exp))
  (variable-environment-type-of-variable exp var-env))

(defun type-of-constant (exp var-env)
  (unless (variable-environment-constant-exists-p exp var-env)
    (error "undefined constant: ~A" exp))
  (variable-environment-type-of-constant exp var-env))

(defun type-of-symbol-macro (exp var-env func-env)
  (unless (variable-environment-symbol-macro-exists-p exp var-env)
    (error "undefined symbol macro: ~A" exp))
  (let ((expansion (variable-environment-symbol-macro-expansion exp var-env)))
    (type-of-expression expansion var-env func-env)))

(defun type-of-symbol (exp var-env func-env)
  (cond
    ((variable-p exp var-env) (type-of-variable exp var-env))
    ((constant-p exp var-env) (type-of-constant exp var-env))
    ((symbol-macro-p exp var-env) (type-of-symbol-macro exp var-env func-env))
    (t (error "undefined variable: ~A" exp))))

(defun type-of-variable-reference (exp var-env func-env)
  (cond ((vector-variable-reference-p exp)
         (type-of-vector-variable-reference exp var-env func-env))
        ((array-variable-reference-p exp)
         (type-of-array-variable-reference exp var-env func-env))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-vector-variable-reference (exp var-env func-env)
  (cl-pattern:match exp
    ((selector exp2)
     (let ((selector-type (vector-type-selector-type selector))
           (exp-type      (type-of-expression exp2 var-env func-env)))
       (unless (eq selector-type exp-type)
         (error "invalid variable reference: ~A" exp))
       (vector-type-base-type exp-type)))
    (_ (error "invalid variable reference: ~A" exp))))

(defun type-of-array-variable-reference (exp var-env func-env)
  (cl-pattern:match exp
    (('aref _) (error "invalid variable reference: ~A" exp))
    (('aref exp2 . idxs)
     (let ((type (type-of-expression exp2 var-env func-env)))
       (unless (= (array-type-dimension type) (length idxs))
         (error "invalid dimension: ~A" exp))
       (remove-star type)))
    (_ (error "invalid variable reference: ~A" exp))))

(defun type-of-inline-if (exp var-env func-env)
  (let ((test-exp (inline-if-test-expression exp))
        (then-exp (inline-if-then-expression exp))
        (else-exp (inline-if-else-expression exp)))
    (let ((test-exp-type (type-of-expression test-exp var-env func-env))
          (then-exp-type (type-of-expression then-exp var-env func-env))
          (else-exp-type (type-of-expression else-exp var-env func-env)))
      (when (not (eq test-exp-type 'bool))
        (error "type of the test part of the inline if expression is not bool: ~A" exp))
      (when (not (eq then-exp-type else-exp-type))
        (error "types of the then part and the else part of the inline if expression are not same: ~A" exp))
      then-exp-type)))

(defun type-of-macro-form (exp var-env func-env)
  (type-of-expression (%expand-macro-1 exp func-env) var-env func-env))

(defun type-of-function (exp var-env func-env)
  (cond ((built-in-function-p exp)
         (type-of-built-in-function exp var-env func-env))
        ((user-function-p exp func-env)
         (type-of-user-function exp func-env))
        (t (error "invalid expression: ~A" exp))))

(defun type-of-built-in-function (exp var-env func-env)
  (built-in-function-return-type exp var-env func-env))

(defun type-of-user-function (exp func-env)
  (let ((operator (function-operator exp)))
    (unless (function-environment-function-exists-p operator func-env)
      (error "undefined function: ~A" operator))
    (function-environment-function-return-type operator func-env)))


;;;
;;; Variable environment
;;;

(defun make-varenv-variable (name type)
  (assert (symbolp name))
  (assert (valid-type-p type))
  (list name :variable type))

(defun varenv-variable-p (elem)
  (cl-pattern:match elem
    ((_ :variable _) t)
    (_ nil)))

(defun varenv-variable-name (elem)
  (cl-pattern:match elem
    ((name :variable _) name)
    (_ (error "invalid variable environment variable: ~A" elem))))

(defun varenv-variable-type (elem)
  (cl-pattern:match elem
    ((_ :variable type) type)
    (_ (error "invalid variable environment variable: ~A" elem))))

(defun make-varenv-constant (name type)
  (assert (symbolp name))
  (assert (valid-type-p type))
  (list name :constant type))

(defun varenv-constant-p (elem)
  (cl-pattern:match elem
    ((_ :constant _) t)
    (_ nil)))

(defun varenv-constant-name (elem)
  (cl-pattern:match elem
    ((name :constant _) name)
    (_ (error "invalid variable environment constant: ~A" elem))))

(defun varenv-constant-type (elem)
  (cl-pattern:match elem
    ((_ :constant type) type)
    (_ (error "invalid variable environment constant: ~A" elem))))

(defun make-varenv-symbol-macro (name expansion)
  (assert (symbolp name))
  (list name :symbol-macro expansion))

(defun varenv-symbol-macro-p (elem)
  (cl-pattern:match elem
    ((_ :symbol-macro _) t)
    (_ nil)))

(defun varenv-symbol-macro-name (elem)
  (cl-pattern:match elem
    ((name :symbol-macro _) name)
    (_ (error "invalid variable environment symbol macro: ~A" elem))))

(defun varenv-symbol-macro-expansion (elem)
  (cl-pattern:match elem
    ((_ :symbol-macro expansion) expansion)
    (_ (error "invalid variable environment symbol macro: ~A" elem))))

(defun varenv-name (elem)
  (cond
    ((varenv-variable-p elem) (varenv-variable-name elem))
    ((varenv-constant-p elem) (varenv-constant-name elem))
    ((varenv-symbol-macro-p elem) (varenv-symbol-macro-name elem))
    (t (error "invalid variable environment element: ~A" elem))))

(defun empty-variable-environment ()
  '())

(defun add-variable-to-variable-environment (name type var-env)
  (let ((elem (make-varenv-variable name type)))
    (cons elem var-env)))

(defun add-constant-to-variable-environment (name type var-env)
  (let ((elem (make-varenv-constant name type)))
    (cons elem var-env)))

(defun add-symbol-macro-to-variable-environment (name expansion var-env)
  (let ((elem (make-varenv-symbol-macro name expansion)))
    (cons elem var-env)))

(defun bulk-add-variable-environment (bindings var-env)
  (reduce #'(lambda (var-env2 binding)
              (cl-pattern:match binding
                ((name :variable type)
                 (add-variable-to-variable-environment name type var-env2))
                ((name :constant type)
                 (add-constant-to-variable-environment name type var-env2))
                ((name :symbol-macro expansion)
                 (add-symbol-macro-to-variable-environment name expansion var-env2))
                (_ (error "invalid variable environment element: ~A" binding))))
          bindings :initial-value var-env))

(defun %add-function-arguments (name def var-env)
  (if name
      (let ((arg-bindings (kernel-definition-function-arguments name def)))
        (reduce #'(lambda (var-env arg-binding)
                    (destructuring-bind (var type) arg-binding
                  (add-variable-to-variable-environment var type var-env)))
                arg-bindings :initial-value var-env))
      var-env))

(defun %add-symbol-macros (def var-env)
  (labels ((%symbol-macro-binding (name)
             (let ((name (kernel-definition-symbol-macro-name name def))
                   (expansion (kernel-definition-symbol-macro-expansion name def)))
               (list name :symbol-macro expansion))))
    (let ((symbol-macro-bindings (mapcar #'%symbol-macro-binding
                                         (kernel-definition-symbol-macro-names def))))
      (bulk-add-variable-environment symbol-macro-bindings var-env))))

(defun %add-constants (def var-env)
  (labels ((%constant-binding (name)
             (let ((name (kernel-definition-constant-name name def))
                   (type (kernel-definition-constant-type name def)))
               (list name :constant type))))
    (let ((constant-bindings (mapcar #'%constant-binding
                                     (kernel-definition-constant-names def))))
      (bulk-add-variable-environment constant-bindings var-env))))

(defun make-variable-environment-with-kernel-definition (name def)
  (%add-function-arguments name def
    (%add-symbol-macros def
      (%add-constants def
        (empty-variable-environment)))))

(defmacro with-variable-environment ((var-env bindings) &body body)
  `(let ((,var-env (bulk-add-variable-environment ',bindings (empty-variable-environment))))
     ,@body))

(defun lookup-variable-environment (name var-env)
  (find name var-env :key #'varenv-name))

(defun variable-environment-variable-exists-p (name var-env)
  (varenv-variable-p (lookup-variable-environment name var-env)))

(defun variable-environment-constant-exists-p (name var-env)
  (varenv-constant-p (lookup-variable-environment name var-env)))

(defun variable-environment-symbol-macro-exists-p (name var-env)
  (varenv-symbol-macro-p (lookup-variable-environment name var-env)))

(defun variable-environment-type-of-variable (name var-env)
  (unless (variable-environment-variable-exists-p name var-env)
    (error "undefined varialbe name: ~A" name))
  (varenv-variable-type (lookup-variable-environment name var-env)))

(defun variable-environment-type-of-constant (name var-env)
  (unless (variable-environment-constant-exists-p name var-env)
    (error "undefined constant name: ~A" name))
  (varenv-constant-type (lookup-variable-environment name var-env)))

(defun variable-environment-symbol-macro-expansion (name var-env)
  (unless (variable-environment-symbol-macro-exists-p name var-env)
    (error "undefined symbol macro name: ~A" name))
  (varenv-symbol-macro-expansion (lookup-variable-environment name var-env)))


;;;
;;; Function environment
;;;

(defun make-funcenv-function (name return-type args body)
  (assert (symbolp name))
  (assert (valid-type-p return-type))
  (assert (listp args))
  (dolist (arg args)
    (assert (= (length arg) 2))
    (assert (symbolp (car arg)))
    (assert (valid-type-p (cadr arg))))
  (assert (listp body))
  (list name :function return-type args body))

(defun funcenv-function-p (elem)
  (cl-pattern:match elem
    ((_ :function _ _ _) t)
    (_ nil)))

(defun funcenv-function-name (elem)
  (cl-pattern:match elem
    ((name :function _ _ _) name)
    (_ (error "invalid function environment function: ~A" elem))))

(defun funcenv-function-c-name (elem)
  (compile-identifier-with-package-name (funcenv-function-name elem)))

(defun funcenv-function-return-type (elem)
  (cl-pattern:match elem
    ((_ :function return-type _ _) return-type)
    (_ (error "invalid function environment function: ~A" elem))))

(defun funcenv-function-arguments (elem)
  (cl-pattern:match elem
    ((_ :function _ arguments _) arguments)
    (_ (error "invalid function environment function: ~A" elem))))

(defun funcenv-function-argument-types (elem)
  (mapcar #'cadr (funcenv-function-arguments elem)))

(defun funcenv-function-body (elem)
  (cl-pattern:match elem
    ((_ :function _ _ body) body)
    (_ (error "invalid function environment function: ~A" elem))))

(defun make-funcenv-macro (name args body expander)
  (assert (symbolp name))
  (assert (listp args))
  (assert (listp body))
  (assert (functionp expander))
  (list name :macro args body expander))

(defun funcenv-macro-p (elem)
  (cl-pattern:match elem
    ((_ :macro _ _ _) t)
    (_ nil)))

(defun funcenv-macro-name (elem)
  (cl-pattern:match elem
    ((name :macro _ _ _) name)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-macro-arguments (elem)
  (cl-pattern:match elem
    ((_ :macro arguments _ _) arguments)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-macro-body (elem)
  (cl-pattern:match elem
    ((_ :macro _ body _) body)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-macro-expander (elem)
  (cl-pattern:match elem
    ((_ :macro _ _ expander) expander)
    (_ (error "invalid function environment macro: ~A" elem))))

(defun funcenv-name (elem)
  (cond
    ((funcenv-function-p elem) (funcenv-function-name elem))
    ((funcenv-macro-p elem) (funcenv-macro-name elem))
    (t (error "invalid function environment element: ~A" elem))))

(defun empty-function-environment ()
  '())

(defun add-function-to-function-environment (name return-type arguments body func-env)
  (let ((elem (make-funcenv-function name return-type arguments body)))
    (cons elem func-env)))

(defun add-macro-to-function-environment (name arguments body expander func-env)
  (let ((elem (make-funcenv-macro name arguments body expander)))
    (cons elem func-env)))

(defun bulk-add-function-environment (bindings func-env)
  (reduce #'(lambda (func-env2 binding)
              (cl-pattern:match binding
                ((name :function return-type args body)
                 (add-function-to-function-environment name return-type args body func-env2))
                ((name :macro args body expander)
                 (add-macro-to-function-environment name args body expander func-env2))
                (_ (error "invalid function environment element: ~A" binding))))
          bindings :initial-value func-env))

(defun make-function-environment-with-kernel-definition (def)
  (labels ((%function-binding (name)
             (let ((name (kernel-definition-function-name name def))
                   (return-type (kernel-definition-function-return-type name def))
                   (args (kernel-definition-function-arguments name def))
                   (body (kernel-definition-function-body name def)))
               (list name :function return-type args body)))
           (%macro-binding (name)
             (let ((name (kernel-definition-macro-name name def))
                   (args (kernel-definition-macro-arguments name def))
                   (body (kernel-definition-macro-body name def))
                   (expander (kernel-definition-macro-expander name def)))
               (list name :macro args body expander))))
    (let ((function-bindings (mapcar #'%function-binding
                                     (kernel-definition-function-names def)))
          (macro-bindings (mapcar #'%macro-binding
                                  (kernel-definition-macro-names def))))
      (bulk-add-function-environment macro-bindings
        (bulk-add-function-environment function-bindings
          (empty-function-environment))))))

(defmacro with-function-environment ((func-env bindings) &body body)
  (labels ((aux (binding)
             (cl-pattern:match binding
               ((name :function return-type args body) `(list ',name :function ',return-type ',args ',body))
               ((name :macro args body) (alexandria:with-gensyms (args0)
                                          `(list ',name :macro ',args ',body
                                                 (lambda (,args0) (destructuring-bind ,args ,args0 ,@body)))))
               (_ `',binding))))
    (let ((bindings2 `(list ,@(mapcar #'aux bindings))))
      `(let ((,func-env (bulk-add-function-environment ,bindings2 (empty-function-environment))))
         ,@body))))

(defun lookup-function-environment (name func-env)
  (find name func-env :key #'funcenv-name))

(defun function-environment-function-exists-p (name func-env)
  (funcenv-function-p (lookup-function-environment name func-env)))

(defun function-environment-macro-exists-p (name func-env)
  (funcenv-macro-p (lookup-function-environment name func-env)))

(defun function-environment-function-c-name (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-c-name (lookup-function-environment name func-env)))

(defun function-environment-function-return-type (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-return-type (lookup-function-environment name func-env)))

(defun function-environment-function-arguments (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-arguments (lookup-function-environment name func-env)))

(defun function-environment-function-argument-types (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-argument-types (lookup-function-environment name func-env)))

(defun function-environment-function-body (name func-env)
  (unless (function-environment-function-exists-p name func-env)
    (error "undefined function name: ~A" name))
  (funcenv-function-body (lookup-function-environment name func-env)))

(defun function-environment-macro-arguments (name func-env)
  (unless (function-environment-macro-exists-p name func-env)
    (error "undefined macro name: ~A" name))
  (funcenv-macro-arguments (lookup-function-environment name func-env)))

(defun function-environment-macro-body (name func-env)
  (unless (function-environment-macro-exists-p name func-env)
    (error "undefined macro name: ~A" name))
  (funcenv-macro-body (lookup-function-environment name func-env)))

(defun function-environment-macro-expander (name func-env)
  (unless (function-environment-macro-exists-p name func-env)
    (error "undefined macro name: ~A" name))
  (funcenv-macro-expander (lookup-function-environment name func-env)))


;;;
;;; Utilities
;;;

(defun compile-identifier (idt)
  (substitute-if #\_ (lambda (char)
                       (and (not (alphanumericp char))
                            (not (char= #\_ char))
                            (not (char= #\* char))))
                 (string-downcase idt)))

(defun compile-identifier-with-package-name (name)
  (let ((package-name (compile-identifier (package-name (symbol-package name))))
        (function-name (compile-identifier name)))
    (concatenate 'string package-name "_" function-name)))

(defun join (str xs &key (remove-nil nil))
  (let ((xs2 (if remove-nil (remove-if #'null xs) xs)))
    (if (not (null xs2))
      (reduce #'(lambda (a b) (concatenate 'string a str b)) xs2)
      "")))

(defun indent (n str)
  (labels ((aux (x)
             (concatenate 'string (spaces n) x)))
    (unlines (mapcar #'aux (lines str)))))

(defun lines (str)
  (split-sequence:split-sequence #\LineFeed str :remove-empty-subseqs t))

(defun unlines (&rest args)
  (cond ((null args) "")
        ((listp (car args)) (join (string #\LineFeed) (car args) :remove-nil t))
        (t (join (string #\LineFeed) args :remove-nil t))))

(defun spaces (n)
  (if (< 0 n)
      (concatenate 'string " " (spaces (1- n)))
      ""))

(defun cl-cuda-symbolicate (&rest args)
  (intern (apply #'concatenate 'string (mapcar #'princ-to-string args))
          :cl-cuda.lang))
