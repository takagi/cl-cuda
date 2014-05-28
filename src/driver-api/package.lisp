#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.driver-api
  (:use :cl)
  (:export ;; Types
           :cu-result
           :cu-device
           :cu-context
           :cu-module
           :cu-function
           :cu-stream
           :cu-event
           :cu-graphics-resource
           :cu-device-ptr
           :size-t
           ;; Enums
           :cu-event-default
           :cu-event-blocking-sync
           :cu-event-disable-timing
           :cu-event-interprocess
           :cu-graphics-register-flags-none
           :cu-graphics-register-flags-read-only
           :cu-graphics-register-flags-write-discard
           :cu-graphics-register-flags-surface-ldst
           :cu-graphics-register-flags-texture-gather
           :cu-graphics-map-resource-flags-none
           :cu-graphics-map-resource-flags-read-only
           :cu-graphics-map-resource-flags-write-discard
           ;; Functions
           :cu-init
           :cu-device-get
           :cu-device-get-count
           :cu-device-compute-capability
           :cu-device-get-name
           :cu-ctx-create
           :cu-gl-ctx-create
           :cu-ctx-destroy
           :cu-ctx-synchronize
           :cu-mem-alloc
           :cu-mem-free
           :cu-memcpy-host-to-device
           :cu-memcpy-device-to-host
           :cu-module-load
           :cu-module-unload
           :cu-module-get-function
           :cu-launch-kernel
           :cu-event-create
           :cu-event-destroy
           :cu-event-elapsed-time
           :cu-event-query
           :cu-event-record
           :cu-event-synchronize
           :cu-graphics-gl-register-buffer
           :cu-graphics-map-resources
           :cu-graphics-resource-get-mapped-pointer
           :cu-graphics-resource-set-map-flags
           :cu-graphics-unmap-resources
           :cu-graphics-unregister-resource
           ;; Messages
           :*show-messages*
           ;; CUcontext
           :create-cu-context
           :destroy-cu-context
           :with-cu-context)
  (:import-from :alexandria
                :ensure-list
                :symbolicate
                :with-gensyms))
