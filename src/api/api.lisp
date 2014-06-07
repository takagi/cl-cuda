#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api
  (:use :cl :cl-reexport))
(in-package :cl-cuda.api)

(import '(cl-cuda.api.nvcc:*tmp-path*
          cl-cuda.api.nvcc:*nvcc-options*
          cl-cuda.api.nvcc:*nvcc-binary*))
(export '(cl-cuda.api.nvcc:*tmp-path*
          cl-cuda.api.nvcc:*nvcc-options*
          cl-cuda.api.nvcc:*nvcc-binary*))
(reexport-from :cl-cuda.api.kernel-manager)
(reexport-from :cl-cuda.api.defkernel)
(reexport-from :cl-cuda.api.context)
(reexport-from :cl-cuda.api.memory)
(reexport-from :cl-cuda.api.timer)
