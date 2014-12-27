#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-interop
  (:use :cl :cl-reexport))
(in-package :cl-cuda-interop)

(reexport-from :cl-cuda-interop.driver-api
               :include '(:*show-messages*
                          :*sdk-not-found*))
(reexport-from :cl-cuda.lang)
(reexport-from :cl-cuda-interop.api)
