#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-interop.api
  (:use :cl :cl-reexport))
(in-package :cl-cuda-interop.api)

(reexport-from :cl-cuda.api
               :exclude '(;; context
                          :create-cuda-context
                          :with-cuda
                          ;; memory
                          :alloc-memory-block
                          :free-memory-block
                          :memory-block-p
                          :memory-block-device-ptr
                          :memory-block-host-ptr
                          :memory-block-type
                          :memory-block-size
                          :with-memory-block
                          :with-memory-blocks
                          :sync-memory-block
                          :memory-block-aref
                          ;; defkernel
                          :defkernel))
(reexport-from :cl-cuda-interop.api.context)
(reexport-from :cl-cuda-interop.api.memory)
(reexport-from :cl-cuda-interop.api.defkernel)
