#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.driver-api
  (:use :cl :cl-test-more
        :cl-cuda.driver-api)
  (:import-from :cl-cuda.driver-api
                :defcuenum
                :enum-keyword
                :enum-value)
  (:import-from :alexandria
                :with-gensyms))
