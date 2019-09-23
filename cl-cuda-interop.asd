#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(defsystem "cl-cuda-interop"
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
  ;; :long-description #.(read-file-string (subpathname *load-pathname* "README.markdown"))
  :in-order-to ((test-op (test-op "cl-cuda-interop-test"))))
