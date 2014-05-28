#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda-test.api
  (:use :cl :cl-test-more
        :cl-cuda.lang
        :cl-cuda.api)
  (:shadowing-import-from :cl-cuda.api
                          :expand-macro
                          :expand-macro-1))
