#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api
  (:use :cl :cl-reexport))
(in-package :cl-cuda.api)

(reexport-from :cl-cuda.api.memory)
(reexport-from :cl-cuda.api.context)
