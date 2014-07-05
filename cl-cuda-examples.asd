#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-examples-asd
  (:use :cl :asdf))
(in-package :cl-cuda-examples-asd)

(defsystem cl-cuda-examples
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cl-cuda
               :cl-test-more
               :imago)
  :components ((:module "examples"
                :components
                ((:file "diffuse0")
                 (:file "diffuse1")
                 ; (:file "shared-memory")
                 (:file "vector-add")
                 (:file "sph"))))
  :perform (load-op :after (op c) (asdf:clear-system c)))
