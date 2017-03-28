#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(defsystem "cl-cuda-interop-examples"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on ("cl-cuda-interop")
  :components ((:module "interop/examples"
                :serial t
                :components
                ((:file "nbody")))))
