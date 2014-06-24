#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-interop-test-asd
  (:use :cl :asdf))
(in-package :cl-cuda-interop-test-asd)

(defsystem cl-cuda-interop-test
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cl-cuda-interop
               :cl-test-more)
  :components ((:module "interop/t"
                :serial t
                :components
                ((:file "cl-cuda-interop"))))
  :perform (load-op :after (op c) (asdf:clear-system c)))
