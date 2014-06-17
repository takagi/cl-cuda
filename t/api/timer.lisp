#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.api.timer
  (:use :cl :cl-test-more
        :cl-cuda.api.timer
        :cl-cuda.api.context))
(in-package :cl-cuda-test.api.timer)

(plan nil)


;;;
;;; test TIMER
;;;

(diag "TIMER")

(with-cuda (0)
  (with-timer (timer)
    ;; start timer
    (start-timer timer)
    ;; sleep
    (sleep 1)
    ;; stop and shnchronize timer
    (stop-timer timer)
    (synchronize-timer timer)
    ;; get elapsed time
    (ok (elapsed-time timer))))


(finalize)
