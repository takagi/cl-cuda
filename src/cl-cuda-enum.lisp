#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-cuda)

;;;
;;; cu-event-flags-enum
;;;

(defcuenum cu-event-flags-enum
  (:cu-event-default #X0)
  (:cu-event-blocking-sync #X1)
  (:cu-event-disable-timing #X2)
  (:cu-event-interprocess #X4));)
