#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(eval-when (:load-toplevel :execute)
  (asdf:operate 'asdf:load-op 'cffi-grovel))

(in-package :cl-user)
(defpackage cl-cuda-interop-asd
  (:use :cl :asdf))
(in-package :cl-cuda-interop-asd)

(defsystem cl-cuda-interop
  :version "0.1"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cl-cuda :cl-opengl :cl-glu :cl-glut)
  :components ((:module "interop/src"
                :serial t
                :components
                ((:module "driver-api"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "type")
                   (:file "enum")
                   (:file "function")))
                 (:module "api"
                  :serial t
                  :components
                  ((:file "memory")
                   (:file "context")
                   (:file "defkernel")
                   (:file "api")))
                 (:file "cl-cuda-interop"))))
  :description "Cl-cuda with OpenGL interoperability."
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
  :in-order-to ((test-op (load-op cl-cuda-interop-test))))
