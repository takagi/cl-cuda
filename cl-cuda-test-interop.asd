#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test-interop-asd
  (:use :cl :asdf))
(in-package :cl-cuda-test-interop-asd)

(defsystem cl-cuda-test-interop
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cl-cuda
               :cl-test-more)
  :components ((:module "t"
                :serial t
                :components
                ((:file "interop/package-interop")
                 (:file "interop/test-cl-cuda-interop"))))
  :perform (load-op :after (op c) (asdf:clear-system c)))
