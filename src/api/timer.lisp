#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.timer
  (:use :cl
        :cl-cuda.driver-api)
  (:export :create-timer
           :destroy-timer
           :start-timer
           :stop-timer
           :synchronize-timer
           :elapsed-time
           :with-timer))
(in-package :cl-cuda.api.timer)


;;;
;;; CUevent
;;;

(defun create-cu-event ()
  (cffi:with-foreign-object (cu-event 'cu-event)
    (cu-event-create cu-event cu-event-default)
    (cffi:mem-ref cu-event 'cu-event)))

(defun destroy-cu-event (cu-event)
  (cu-event-destroy cu-event))

(defun record-cu-event (cu-event)
  (cu-event-record cu-event (cffi:null-pointer)))

(defun sync-cu-event (cu-event)
  (cu-event-synchronize cu-event))

(defun %elapsed-time (start stop)
  (cffi:with-foreign-object (msec :float)
    (cu-event-elapsed-time msec start stop)
    (cffi:mem-ref msec :float)))


;;;
;;; Timer
;;;

(defstruct (timer (:constructor %make-timer))
  (start-event nil :read-only t)
  (stop-event nil :read-only t))

(defun create-timer ()
  (let ((start (create-cu-event))
        (stop (create-cu-event)))
    (%make-timer :start-event start :stop-event stop)))

(defun destroy-timer (timer)
  (destroy-cu-event (timer-start-event timer))
  (destroy-cu-event (timer-stop-event timer)))

(defun start-timer (timer)
  (record-cu-event (timer-start-event timer)))

(defun stop-timer (timer)
  (record-cu-event (timer-stop-event timer)))

(defun synchronize-timer (timer)
  (sync-cu-event (timer-stop-event timer)))

(defun elapsed-time (timer)
  (let ((start (timer-start-event timer))
        (stop (timer-stop-event timer)))
    (%elapsed-time start stop)))

(defmacro with-timer ((var) &body body)
  `(let ((,var (create-timer)))
     (unwind-protect (progn ,@body)
       (destroy-timer ,var))))
