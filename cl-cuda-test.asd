#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test-asd
  (:use :cl :asdf))
(in-package :cl-cuda-test-asd)

(defsystem cl-cuda-test
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:cl-cuda
               :cl-test-more)
  :components ((:module "t"
                :serial t
                :components
                ((:module "driver-api"
                  :serial t
                  :components
                  ((:file "driver-api")))
                 (:module "lang"
                  :serial t
                  :components
                  ((:file "util")
                   (:file "data")
                   (:file "type")
                   (:file "syntax")
                   (:file "environment")
                   (:file "built-in")
                   (:file "kernel")
                   (:file "compiler/compile-data")
                   (:file "compiler/compile-type")
                   (:file "compiler/type-of-expression")
                   (:file "compiler/compile-expression")
                   (:file "compiler/compile-statement")
                   (:file "compiler/compile-kernel")))
                 (:module "api"
                  :serial t
                  :components
                  ((:file "kernel-manager")
                   (:file "memory")
                   (:file "defkernel")
                   (:file "timer"))))))
  :perform (load-op :after (op c) (asdf:clear-system c)))
