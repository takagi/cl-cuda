#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

#|
  Author: Masayuki Takagi (kamonama@gmail.com)
|#

(eval-when (:load-toplevel :execute)
  (asdf:operate 'asdf:load-op 'cffi-grovel))

(in-package :cl-user)
(defpackage cl-cuda-asd
  (:use :cl :asdf))
(in-package :cl-cuda-asd)

(defsystem cl-cuda
  :version "0.1"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cffi :alexandria :anaphora :external-program :osicat
               :cl-pattern :split-sequence :cl-opengl #|:cl-glut|#)
  :components ((:module "src"
                :serial t
                :components
                ((:file "package")
                 (cffi-grovel:grovel-file "grovel")
                 (:file "cl-cuda-error-string")
                 (:file "cl-cuda"))))
  :description ""
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.markdown"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (load-op cl-cuda-test))))
