#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(defsystem "cl-cuda-test"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on ("cl-cuda"
               "prove")
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
                   (:file "timer")))))))
