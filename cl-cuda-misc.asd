#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(defsystem "cl-cuda-misc"
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on ("local-time" "cl-emb")
  :components ((:module "misc"
                :serial t
                :components
                ((:file "package")
                 (:file "convert-error-string")))))
