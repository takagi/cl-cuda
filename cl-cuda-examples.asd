#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(defsystem "cl-cuda-examples"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on ("cl-cuda"
               "imago")
  :components ((:module "examples"
                :components
                ((:file "diffuse0")
                 (:file "diffuse1")
                 ; (:file "shared-memory")
                 (:file "vector-add")
                 (:file "defglobal")
                 (:file "sph")
                 (:file "sph-cpu")))))
