#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-misc-asd
  (:use :cl :asdf))
(in-package :cl-cuda-misc-asd)

(defsystem cl-cuda-misc
  :author "Masayuki Takagi"
  :license "LLGPL"
  :depends-on (:local-time :cl-emb)
  :components ((:module "misc"
                :serial t
                :components
                ((:file "package")
                 (:file "convert-error-string"))))
  :perform (load-op :after (op c) (asdf:clear-system c)))
