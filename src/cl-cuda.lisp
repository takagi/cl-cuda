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
      (defpackage cl-cuda
        (:use :cl :cl-reexport)))))
(in-package :cl-cuda)

(reexport-from :cl-cuda.driver-api
               :include '(:*show-messages*
                          :*sdk-not-found*))
(reexport-from :cl-cuda.lang)
(reexport-from :cl-cuda.api)
