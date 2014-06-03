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
           :with-timer)
  (:shadow :elapsed-time))
(in-package :cl-cuda.api.timer)


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
    (cl-cuda.driver-api:elapsed-time start stop)))

(defmacro with-timer ((var) &body body)
  `(let ((,var (create-timer)))
     (unwind-protect (progn ,@body)
       (destroy-timer ,var))))
