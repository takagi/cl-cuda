#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (locally
      (declare #+sbcl
               (sb-ext:muffle-conditions sb-kernel::package-at-variance))
    (handler-bind
        (#+sbcl (sb-kernel::package-at-variance #'muffle-warning))
      (defpackage cl-cuda.api
        (:use :cl :cl-reexport)))))
(in-package :cl-cuda.api)

(reexport-from :cl-cuda.api.nvcc
               :include '(:*tmp-path*
                          :*nvcc-options*
                          :*nvcc-binary*))
(reexport-from :cl-cuda.api.context)
(reexport-from :cl-cuda.api.memory)
(reexport-from :cl-cuda.api.defkernel)
(reexport-from :cl-cuda.api.macro)
(reexport-from :cl-cuda.api.timer)

;; reexport no symbols from cl-cuda.api.kernel-manager package
