#|
  This file is a part of cl-cuda project.
  Copyright (c) 2014 Masayuki Takagi (kamonama@gmail.com)
|#


(in-package :cl-cuda.driver-api)


;;;
;;; Types
;;;

(cffi:defctype cu-result :unsigned-int)
(cffi:defctype cu-device :int)
(cffi:defctype cu-context :pointer)
(cffi:defctype cu-module :pointer)
(cffi:defctype cu-function :pointer)
(cffi:defctype cu-stream :pointer)
(cffi:defctype cu-event :pointer)
(cffi:defctype cu-graphics-resource :pointer)

;; The followings are just place holders and should be rederined in grovel if
;; cuda.h found.
(cffi:defctype cu-device-ptr :unsigned-long-long)
(cffi:defctype size-t :unsigned-long)