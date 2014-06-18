#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-interop-examples-asd
  (:use :cl :asdf))
(in-package :cl-cuda-interop-examples-asd)

(defsystem cl-cuda-interop-examples
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cl-cuda-interop
               :cl-test-more)
  :components ((:module "interop/examples"
                :serial t
                :components
                ((:file "nbody"))))
  :perform (load-op :after (op c) (asdf:clear-system c)))
