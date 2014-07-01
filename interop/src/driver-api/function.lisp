#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda-interop.driver-api)


;;;
;;; Functions
;;;

;; cuGLCtxCreate
(defcufun (cu-gl-ctx-create "cuGLCtxCreate_v2") cu-result
  (pctx (:pointer cu-context))
  (flags :unsigned-int)
  (dev cu-device))

;; cuGraphicsGLRegisterBuffer
(defcufun (cu-graphics-gl-register-buffer "cuGraphicsGLRegisterBuffer") cu-result
  (p-cuda-resource (:pointer cu-graphics-resource))
  (buffer %gl:uint)
  (flags :unsigned-int))

;; cuGraphicsMapResources
(defcufun (cu-graphics-map-resources "cuGraphicsMapResources") cu-result
  (count     :unsigned-int)
  (resources (:pointer cu-graphics-resource))
  (hstream   cu-stream))

;; cuGraphicsResourceGetMappedPointer
(defcufun (cu-graphics-resource-get-mapped-pointer "cuGraphicsResourceGetMappedPointer_v2") cu-result
  (pdevptr  (:pointer cu-device-ptr))
  (psize    (:pointer size-t))
  (resource cu-graphics-resource))

;; cuGraphicsResourceSetMapFlags
(defcufun (cu-graphics-resource-set-map-flags "cuGraphicsResourceSetMapFlags") cu-result
  (resource cu-graphics-resource)
  (flags    :unsigned-int))

;; cuGraphicsUnmapResources
(defcufun (cu-graphics-unmap-resources "cuGraphicsUnmapResources") cu-result
  (count     :unsigned-int)
  (resources (:pointer cu-graphics-resource))
  (hstream   cu-stream))

;; cuGraphicsUnregisterResource
(defcufun (cu-graphics-unregister-resource "cuGraphicsUnregisterResource") cu-result
  (resource cu-graphics-resource))
