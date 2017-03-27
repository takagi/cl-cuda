#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(defsystem "cl-cuda-interop-test"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on ("cl-cuda-interop" "prove")
  :components ((:module "interop/t"
                :serial t
                :components
                ((:file "cl-cuda-interop")))))
