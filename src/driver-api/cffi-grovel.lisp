#|
  This file is a part of cl-cuda project.
  Copyright (c) 2014 Masayuki Takagi (kamonama@gmail.com)
|#

#|
  This file is a work around for CFFI Bug#1272009
|#


(in-package :cffi-grovel)


;; same as grovel syntax CONSTANT except C preprocessor conditional
(define-grovel-syntax constant-from-enum ((lisp-name &rest c-names)
                                          &key (type 'integer) documentation
                                               optional)
  (when (keywordp lisp-name)
    (setf lisp-name (format-symbol "~A" lisp-name)))
  (c-section-header out "constant" lisp-name)
  (dolist (c-name c-names)
;;     (format out "~&#ifdef ~A~%" c-name)
    (c-export out lisp-name)
    (c-format out "(cl:defconstant ")
    (c-print-symbol out lisp-name t)
    (c-format out " ")
    (ecase type
      (integer
       (format out "~&  if(_64_BIT_VALUE_FITS_SIGNED_P(~A))~%" c-name)
       (format out "    fprintf(output, \"%lli\", (int64_t) ~A);" c-name)
       (format out "~&  else~%")
       (format out "    fprintf(output, \"%llu\", (uint64_t) ~A);" c-name))
      (double-float
       (format out "~&  fprintf(output, \"%s\", print_double_for_lisp((double)~A));~%" c-name)))
    (when documentation
      (c-format out " ~S" documentation))
    (c-format out ")~%")
;;     (format out "~&#else~%"))
;;   (unless optional
;;     (c-format out "(cl:warn 'cffi-grovel:missing-definition :name '~A)~%"
;;               lisp-name))
;;   (dotimes (i (length c-names))
;;     (format out "~&#endif~%")))
    ))

;; redeine DEFINE-CONSTANTS-FROM-ENUM to call method for CONSTANT-FROM-ENUM 
(defun define-constants-from-enum (out enum-list)
  (dolist (enum enum-list)
    (destructuring-bind ((lisp-name &rest c-names) &rest options)
        enum
      (%process-grovel-form
       'constant-from-enum out
       `((,(intern (string lisp-name)) ,(car c-names))
         ,@options)))))
