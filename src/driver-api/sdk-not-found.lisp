#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#


(in-package cl-cuda.driver-api)

(defvar *sdk-not-found* nil)

(define-condition sdk-not-found-error (simple-error) ()
  (:report "CUDA SDK not found."))
