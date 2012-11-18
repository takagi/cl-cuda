#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

#|
  This file is based on the CUDA SDK's "nbody" sample.
|#

(in-package :cl-user)
(defpackage cl-cuda-examples.nbody
  (:use :cl
        :cl-cuda)
  (:export :main
           :*gpu*))
(in-package :cl-cuda-examples.nbody)


(defvar *gpu* t)


;;;
;;; Vec3 array
;;;

(deftype vec3-array () '(simple-array single-float (*)))

(defun make-vec3-array (n)
  (let ((size (* n 3)))
    (make-array (list size) :element-type 'single-float :initial-element 0.0)))

(defun vec3-array-dimension (ary)
  (/ (array-dimension ary 0) 3))

(defun vec3-aref (ary i elm)
  (ecase elm
    (:x (aref ary (the fixnum (+ (the fixnum (* i 3)) 0))))
    (:y (aref ary (the fixnum (+ (the fixnum (* i 3)) 1))))
    (:z (aref ary (the fixnum (+ (the fixnum (* i 3)) 2))))))

(defun (setf vec3-aref) (val ary i elm)
  (ecase elm
    (:x (setf (aref ary (the fixnum (+ (the fixnum (* i 3)) 0))) val))
    (:y (setf (aref ary (the fixnum (+ (the fixnum (* i 3)) 1))) val))
    (:z (setf (aref ary (the fixnum (+ (the fixnum (* i 3)) 2))) val))))

(defmacro with-vec3-array-values (((x y z) (ary i)) &body body)
  ;; for fast read
  `(multiple-value-bind (,x ,y ,z) (vec3-array-values ,ary ,i)
     (declare (type single-float ,x ,y ,z))
     ,@body))

(declaim (inline vec3-array-values))
(defun vec3-array-values (ary i)
  (declare (optimize (speed 3) (safety 0)))
  (declare (type vec3-array ary)
           (type fixnum i))
  (values (aref ary (the fixnum (+ (the fixnum (* i 3)) 0)))
          (aref ary (the fixnum (+ (the fixnum (* i 3)) 1)))
          (aref ary (the fixnum (+ (the fixnum (* i 3)) 2)))))

(defmacro set-vec3-array ((ary i) x y z)
  ;; for fast write
  `(%set-vec3-array ,ary ,i ,x ,y ,z))

(declaim (inline %set-vec3-array))
(defun %set-vec3-array (ary i x y z)
  (declare (optimize (speed 3) (safety 0)))
  (declare (type vec3-array ary)
           (type fixnum i))
  (setf (aref ary (the fixnum (+ (the fixnum (* i 3)) 0))) x
        (aref ary (the fixnum (+ (the fixnum (* i 3)) 1))) y
        (aref ary (the fixnum (+ (the fixnum (* i 3)) 2))) z))


;;;
;;; Abstract array structure
;;;

(defun alloc-array (n gpu-p)
  (if gpu-p
      (list :gpu (alloc-memory-block 'float4 n))
      (list :cpu (make-vec3-array n))))

(defun free-array (ary)
  (ecase (array-type ary)
    (:gpu (progn
            (free-memory-block (cadr ary))
            (setf (cadr ary) nil)))
    (:cpu nil)))

(defun array-ref (ary i)
  (destructuring-bind (type raw-ary) ary
    (ecase type
      (:gpu (let ((x (mem-aref raw-ary i)))
              (values (float4-x x) (float4-y x) (float4-z x) (float4-w x))))
      (:cpu (values (vec3-aref raw-ary i :x)
                    (vec3-aref raw-ary i :y)
                    (vec3-aref raw-ary i :z)
                    1.0)))))

(defun (setf array-ref) (val ary i)
  (destructuring-bind (x y z w) val
    (destructuring-bind (type raw-ary) ary
      (ecase type
        (:gpu (setf (mem-aref raw-ary i) (make-float4 x y z w)))
        (:cpu (setf (vec3-aref raw-ary i :x) x
                    (vec3-aref raw-ary i :y) y
                    (vec3-aref raw-ary i :z) z))))))

(defun array-type (ary)
  (car ary))

(defun raw-array (ary)
  (cadr ary))

(defun memcpy-array-host-to-device (ary)
  (ecase (array-type ary)
    (:gpu (memcpy-host-to-device (raw-array ary)))
    (:cpu (error "invalid array type: cpu"))))

(defun memcpy-array-device-to-host (ary)
  (ecase (array-type ary)
    (:gpu (memcpy-device-to-host (raw-array ary)))
    (:cpu (error "invalid array type: cpu"))))


;;;
;;; Kernel functions
;;;

(defkernel body-body-interaction (float3 ((ai float3) (bi float4) (bj float4)))
  (let ((r (float3 (- (float4-x bj) (float4-x bi))
                   (- (float4-y bj) (float4-y bi))
                   (- (float4-z bj) (float4-z bi))))
        (softening-squared (* 0.1 0.1))
        (dist-sqr (+ (* (float3-x r) (float3-x r))
                     (* (float3-y r) (float3-y r))
                     (* (float3-z r) (float3-z r))
                     softening-squared))
        (inv-dist (rsqrtf dist-sqr))
        (inv-dist-cube (* inv-dist inv-dist inv-dist))
        (s (* (float4-w bj) inv-dist-cube)))
    (set (float3-x ai) (+ (float3-x ai) (* (float3-x r) s)))
    (set (float3-y ai) (+ (float3-y ai) (* (float3-y r) s)))
    (set (float3-z ai) (+ (float3-z ai) (* (float3-z r) s)))
    (return ai)))

(defkernel gravitation (float3 ((ipos float4) (accel float3) (shared-pos float4*)))
;  (with-shared-memory ((shared-pos float4 256))
  (do ((j 0 (+ j 1)))
      ((>= j block-dim-x))
    (set accel (body-body-interaction accel ipos (aref shared-pos j))))
  (return accel))

(defkernel wrap (int ((x int) (m int)))
  (if (< x m)
      (return x)
      (return (- x m))))

(defkernel compute-body-accel (float3 ((body-pos float4) (positions float4*)
                                       (num-bodies int)))
  (with-shared-memory ((shared-pos float4 256))
    (let ((acc (float3 0.0 0.0 0.0))
          (p block-dim-x)
          (n num-bodies)
          (num-tiles (/ n p)))
      (do ((tile 0 (+ tile 1)))
          ((>= tile num-tiles))
        (let ((idx (+ (* (wrap (+ block-idx-x tile) grid-dim-x) p)
                      thread-idx-x)))
          (set (aref shared-pos thread-idx-x) (aref positions idx)))
        (syncthreads)
        (set acc (gravitation body-pos acc shared-pos))
        (syncthreads))
      (return acc))))

(defkernel compute-body-accel-without-shared-memory (float3 ((body-pos float4) (positions float4*) (num-bodies cl-cuda:int)))
  (let ((acc (float3 0.0 0.0 0.0)))
    (do ((i 0 (+ i 1)))
        ((>= i num-bodies))
      (set acc (body-body-interaction acc body-pos (aref positions i))))
    (syncthreads)
    (return acc)))

(defkernel integrate-bodies (void ((new-pos float4*) (old-pos float4*) (vel float4*)
                                   (delta-time float) (damping float)
                                   (total-num-bodies int)))
  (let ((index (+ (* block-idx-x block-dim-x) thread-idx-x)))
    (if (>= index total-num-bodies)
        (return))
    (let ((position (aref old-pos index))
          (accel (compute-body-accel position old-pos total-num-bodies))
;          (accel (compute-body-accel-without-shared-memory position old-pos total-num-bodies))
          (velocity (aref vel index)))
      (set (float4-x velocity) (+ (float4-x velocity)
                                  (* (float3-x accel) delta-time)))
      (set (float4-y velocity) (+ (float4-y velocity)
                                  (* (float3-y accel) delta-time)))
      (set (float4-z velocity) (+ (float4-z velocity)
                                  (* (float3-z accel) delta-time)))
      
      (set (float4-x velocity) (* (float4-x velocity) damping))
      (set (float4-y velocity) (* (float4-y velocity) damping))
      (set (float4-z velocity) (* (float4-z velocity) damping))
      
      (set (float4-x position) (+ (float4-x position)
                                  (* (float4-x velocity) delta-time)))
      (set (float4-y position) (+ (float4-y position)
                                  (* (float4-y velocity) delta-time)))
      (set (float4-z position) (+ (float4-z position)
                                  (* (float4-z velocity) delta-time)))
      
      (set (aref new-pos index) position)
      (set (aref vel index) velocity))))

(defun integrate-nbody-system-gpu (new-pos old-pos vel delta-time damping num-bodies p)
  (let ((grid-dim  (list (ceiling (/ num-bodies p)) 1 1))
        (block-dim (list p 1 1)))
    (integrate-bodies (raw-array new-pos) (raw-array old-pos) (raw-array vel)
                      delta-time damping num-bodies
                      :grid-dim grid-dim
                      :block-dim block-dim)
    (synchronize-context)))

(declaim (inline body-body-interaction-cpu))
(defun body-body-interaction-cpu (x1 y1 z1 x2 y2 z2)
  (declare (optimize (speed 3) (safety 0)))
  (declare (type single-float x1 y1 z1 x2 y2 z2))
  (let* ((rx                (- x2 x1))
         (ry                (- y2 y1))
         (rz                (- z2 z1))
         (softening-squared (* 0.1 0.1))
         (dist-sqr          (+ (* rx rx) (* ry ry) (* rz rz)
                               softening-squared))
         (inv-dist          (/ 1.0 (sqrt dist-sqr)))
         (inv-dist-cube     (* inv-dist inv-dist inv-dist))
         (w                 1.0)
         (s                 (* w inv-dist-cube)))
    (values (* rx s) (* ry s) (* rz s))))

(defun integrate-bodies-cpu (new-pos old-pos vel delta-time damping total-num-bodies)
  (declare (optimize (speed 3) (safety 0)))
  (declare (type fixnum total-num-bodies)
           (type single-float delta-time damping)
           (type vec3-array new-pos old-pos vel))
  (dotimes (i total-num-bodies)
    (with-vec3-array-values ((x1 y1 z1) (old-pos i))
      (dotimes (j total-num-bodies)
        (with-vec3-array-values ((x2 y2 z2) (old-pos j))
          (when (/= i j)
            (multiple-value-bind (ax ay az) (body-body-interaction-cpu x1 y1 z1 x2 y2 z2)
              (declare (type single-float ax ay az))
              (let ((k (* delta-time damping)))
                (with-vec3-array-values ((vx vy vz) (vel i))
                  (set-vec3-array (vel i) (+ vx (* ax k))
                                          (+ vy (* ay k))
                                          (+ vz (* az k)))))))))
      (with-vec3-array-values ((vx vy vz) (vel i))
        (set-vec3-array (new-pos i) (+ x1 (* vx delta-time))
                                    (+ y1 (* vy delta-time))
                                    (+ z1 (* vz delta-time)))))))

(defun integrate-nbody-system-cpu (new-pos old-pos vel delta-time damping num-bodies)
  (integrate-bodies-cpu (raw-array new-pos) (raw-array old-pos) (raw-array vel)
                        delta-time damping num-bodies))


;;;
;;; cl-glut window subclass
;;;

(defclass nbody-window (glut:window)
  ((nbody-demo :initarg :nbody-demo)
   (counter    :initarg :counter))
  (:default-initargs :width 640 :height 480 :pos-x 100 :pos-y 100
                     :mode '(:double :rgb) :title "nbody"))


;;;
;;; cl-glut event handlers
;;;

(defmethod glut:display-window :before ((w nbody-window))
  (gl:enable :depth-test)
  (gl:clear-color 0 0 0 0))

;(defmethod glut:close ((w nbody-window))
;  nil)

(defmethod glut:display ((w nbody-window))
  (with-slots (nbody-demo counter) w
    ;; update simulation
    (update-nbody-demo nbody-demo)
    ;; clear buffers
    (gl:clear :color-buffer :depth-buffer-bit)
    ;; view transform
    (gl:matrix-mode :modelview)
    (gl:load-identity)
    (gl:translate 0.0 0.0 -100.0)
    (gl:rotate 0.0 1.0 0.0 0.0)
    (gl:rotate 0.0 0.0 1.0 0.0)
    ;; display bodies
    (display-nbody-demo nbody-demo)
    ;; swap buffers
    (glut:swap-buffers)
    ;; display frame rate
    (measure-frame-rate counter)
    (display-frame-rate (nbody-demo-num-bodies nbody-demo)
                        (get-frame-rate counter))))

(defmethod glut:reshape ((w nbody-window) width height)
  ;; configure on projection mode
  (gl:matrix-mode :projection)
  (gl:load-identity)
  (glu:perspective 60.0 (/ width height) 0.1 1000.0)
  ;; configure on model-view mode
  (gl:matrix-mode :modelview)
  (gl:viewport 0 0 width height))

(defmethod glut:idle ((w nbody-window))
  (glut:post-redisplay))


;;;
;;; Performance displayment functions
;;;

(defvar *flops-per-interaction* 20)

(defun compute-perf-stats (num-bodies fps iterations)
  (let* ((interactions-per-second (/ (* num-bodies num-bodies iterations)
                                     (/ 1.0 fps)
                                     1.0e9))
         (gflops                  (* interactions-per-second *flops-per-interaction*)))
    (values interactions-per-second gflops)))

(let ((fps-count 0)
      (fps-limit 5)
      (template "CUDA N-Body (~A bodies): ~,1F fps | ~,1F BIPS | ~,1F GFLOP/s | single precision~%"))
  (defun display-frame-rate (num-bodies fps)
    (incf fps-count)
    (when (>= fps-count fps-limit)
      (multiple-value-bind (interactions-per-second gflops)
          (compute-perf-stats num-bodies fps 1)
        (format t template num-bodies fps interactions-per-second gflops)
        (glut:set-window-title (format nil template num-bodies fps interactions-per-second gflops)))
      (setf fps-count 0
            fps-limit (max fps 1.0)))))


;;;
;;; Frame rate counter
;;;

(defstruct (frame-rate-counter :conc-name)
  timer
  (fps-count 0)
  (fps-limit 5)
  (fps       0))

(defmacro with-frame-rate-counter ((var) &body body)
  `(let (,var)
     (unwind-protect
          (progn (setf ,var (init-frame-rate-counter))
                 ,@body)
       (release-frame-rate-counter ,var))))

(defun init-frame-rate-counter ()
  (let ((counter (make-frame-rate-counter)))
    (setf (timer counter) (create-timer))
    (start-timer (timer counter))
    counter))

(defun release-frame-rate-counter (counter)
  (destroy-timer (timer counter)))

(defun get-frame-rate (counter)
  (fps counter))

(defun measure-frame-rate (counter)
  (symbol-macrolet ((timer     (timer counter))
                    (fps-count (fps-count counter))
                    (fps-limit (fps-limit counter))
                    (fps       (fps counter)))
    (incf fps-count)
    (when (>= fps-count fps-limit)
      (let ((milliseconds (/ (get-elapsed-time timer) fps-count)))
        (setf fps       (/ 1.0 (/ milliseconds 1000.0))
              fps-count 0
              fps-limit (max fps 1.0))))))


;;;
;;; NBody Demo
;;;

(defun unlines (&rest string-list)
  "Concatenates a list of strings and puts newlines between the elements."
  (format nil "~{~A~%~}" string-list))

(defparameter +vertex-shader+
  (unlines "void main()                                                            "
           "{                                                                      "
           "    float pointSize = 500.0 * gl_Point.size;                           "
           "    vec4 vert = gl_Vertex;                                             "
           "    vert.w = 1.0;                                                      "
           "    vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);                   "
           "    gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));            "
           "    gl_TexCoord[0] = gl_MultiTexCoord0;                                "
           ;"    gl_TexCoord[1] = gl_MultiTexCoord1;                                "
           "    gl_Position = ftransform();                                        "
           "    gl_FrontColor = gl_Color;                                          "
           "    gl_FrontSecondaryColor = gl_SecondaryColor;                        "
           "}                                                                      "))

(defparameter +pixel-shader+
  (unlines "uniform sampler2D splatTexture;                                        "
           "void main()                                                            "
           "{                                                                      "
           "    vec4 color2 = gl_SecondaryColor;                                   "
           "    vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st);"
           "    gl_FragColor =                                                     "
           "         color * color2;" ;mix(vec4(0.1, 0.0, 0.0, color.w), color2, color.w);"
           "}                                                                      "))

(defstruct (nbody-demo :conc-name)
  system renderer)

(defmacro with-nbody-demo ((var num-bodies gpu-p) &body body)
  `(let (,var)
     (unwind-protect
        (progn
          (setf ,var (init-nbody-demo ,num-bodies ,gpu-p))
          ,@body)
       (release-nbody-demo ,var))))

(defun init-nbody-demo (num-bodies gpu-p)
  (let ((demo (make-nbody-demo)))
    (setf (system   demo) (init-body-sysmem num-bodies gpu-p)
          (renderer demo) (init-particle-renderer))
    demo))

(defun release-nbody-demo (demo)
  (let ((renderer (renderer demo))
        (system   (system demo)))
    (release-particle-renderer renderer)
    (release-body-system system)))

(defun update-nbody-demo (demo)
  (update-body-system (system demo)))

(defun display-nbody-demo (demo)
  (let ((renderer   (renderer demo))
        (system     (system   demo)))
    (let ((pos        (old-pos system))
          (num-bodies (num-bodies system)))
      (display-particle-renderer renderer pos num-bodies :particle-sprites))))

(defun nbody-demo-num-bodies (demo)
  (num-bodies (system demo)))


;;;
;;; Body System
;;;

(defstruct (body-system :conc-name)
  ;; CUDA device ID
  (dev-id         0     :read-only t)
  ;; memory blocks
  new-pos old-pos vel
  ;; simulation parameters
  (num-bodies     0)
  (delta-time     0.016 :read-only t)
  (damping        1.0   :read-only t)
  (p              256   :read-only t)
  (cluster-scale  1.56  :read-only t)
  (velocity-scale 2.64  :read-only t))

(defun init-body-sysmem (num-bodies gpu-p)
  (let ((system (make-body-system)))
    (setf (num-bodies system) num-bodies)
    (init-cuda-context 0)
    (init-memory-blocks system gpu-p)
    (reset-body-system system)
    (when gpu-p
      (memcpy-array-host-to-device (old-pos system))
      (memcpy-array-host-to-device (vel system)))
    system))

(defun init-memory-blocks (system gpu-p)
  (symbol-macrolet ((new-pos    (new-pos system))
                    (old-pos    (old-pos system))
                    (vel        (vel system))
                    (num-bodies (num-bodies system)))
    (setf new-pos (alloc-array num-bodies gpu-p)
          old-pos (alloc-array num-bodies gpu-p)
          vel     (alloc-array num-bodies gpu-p))))

(defun release-body-system (system)
  (release-memory-blocks system)
  (release-cuda-context))

(defun release-memory-blocks (system)
  (free-array (vel     system))
  (free-array (old-pos system))
  (free-array (new-pos system)))

(defun reset-body-system (system)
  (let ((old-pos        (old-pos system))
        (vel            (vel system))
        (cluster-scale  (cluster-scale system))
        (velocity-scale (velocity-scale system))
        (num-bodies     (num-bodies system)))
    (randomize-bodies old-pos vel cluster-scale velocity-scale num-bodies)))

(defun update-body-system (system)
  (symbol-macrolet ((new-pos    (new-pos system))
                    (old-pos    (old-pos system))
                    (vel        (vel system))
                    (delta-time (delta-time system))
                    (damping    (damping system))
                    (num-bodies (num-bodies system))
                    (p          (p system)))
    (ecase (array-type new-pos)
      (:cpu (integrate-nbody-system-cpu new-pos old-pos vel delta-time damping num-bodies))
      (:gpu (integrate-nbody-system-gpu new-pos old-pos vel delta-time damping num-bodies p)))
    (rotatef new-pos old-pos)))


;;;
;;; Particle renderer
;;;

(defstruct (particle-renderer :conc-name)
  ;; for rendering
  (point-size  1.0                :read-only t)
  (sprite-size 2.0                :read-only t)
  (base-color  #(1.0 0.6 0.3 1.0) :read-only t)
  ;; for shaders
  program vertex-shader pixel-shader
  ;; for texture
  texture texture-data)

(defun init-particle-renderer ()
  (let ((renderer (make-particle-renderer)))
    (init-shaders renderer)
    (create-texture renderer 32)
    renderer))

(defun init-shaders (renderer)
  (symbol-macrolet ((program       (program       renderer))
                    (vertex-shader (vertex-shader renderer))
                    (pixel-shader  (pixel-shader  renderer)))
    ;; create shader objects
    (setf vertex-shader (gl:create-shader :vertex-shader)
          pixel-shader  (gl:create-shader :fragment-shader))
    ;; set shader source codes
    (gl:shader-source vertex-shader +vertex-shader+)
    (gl:shader-source pixel-shader  +pixel-shader+)
    ;; compile source codes
    (gl:compile-shader vertex-shader)
    (gl:compile-shader pixel-shader)
    ;; create an empty program object
    (setf program (gl:create-program))
    ;; attach shader objects to program object
    (gl:attach-shader program vertex-shader)
    (gl:attach-shader program pixel-shader)
    ;; link program object
    (gl:link-program program)))

(defun create-texture (renderer resolution)
  (symbol-macrolet ((texture-data (texture-data renderer))
                    (texture      (texture      renderer)))
    ;; prepare texture ata
    (setf texture-data (cffi:foreign-alloc :unsigned-char :count (* 4 resolution resolution)))
    (create-gaussian-map texture-data resolution)
    ;; generate texture
    (setf texture (first (gl:gen-textures 1)))
    (gl:bind-texture :texture-2d texture)
    (gl:tex-parameter :texture-2d :generate-mipmap :true)
    (gl:tex-parameter :texture-2d :texture-min-filter :linear-mipmap-linear)
    (gl:tex-parameter :texture-2d :texture-mag-filter :linear)
    (gl:tex-image-2d :texture-2d 0 :rgba8 resolution resolution 0
                     :rgba :unsigned-byte texture-data)))

(defun release-particle-renderer (renderer)
  (destroy-texture renderer)
  (release-shaders renderer))

(defun release-shaders (renderer)
  (declare (ignorable renderer))
  (values))

(defun destroy-texture (renderer)
  (cffi:foreign-free (texture-data renderer)))

(defun display-particle-renderer (renderer pos num-bodies mode)
  (let ((point-size  (point-size  renderer))
        (sprite-size (sprite-size renderer))
        (program     (program     renderer))
        (texture     (texture     renderer))
        (base-color  (base-color  renderer)))
    (ecase mode
      (:particle-points
        (gl:color 1.0 1.0 1.0)
        (gl:point-size point-size)
        (draw-points pos num-bodies))
      (:particle-sprites
        ;; enable features
        (gl:enable :point-sprite)
        (gl:tex-env :point-sprite :coord-replace :true)
        (gl:enable :vertex-program-point-size-nv)
        (gl:point-size sprite-size)
        (gl:blend-func :src-alpha :one)
        (gl:enable :blend)
        (gl:depth-mask :false)
        ;; use shader program
        (gl:use-program program)
        (gl:uniformi (gl:get-uniform-location program "splatTexture") 0)
        ;; bind texture
        (gl:active-texture :texture0-arb)
        (gl:bind-texture :texture-2d texture)
        ;; set color
        (gl:color 1.0 1.0 1.0)
        (gl:secondary-color (aref base-color 0) (aref base-color 1) (aref base-color 2))
        ;; draw points
        (draw-points pos num-bodies)
        ;; dont use shader program
        (gl:use-program 0)
        ;; disable features
        (gl:disable :point-sprite-arb)
        (gl:disable :blend)
        (gl:depth-mask :true)))))

(defun draw-points (pos num-bodies)
  (ecase (array-type pos)
    (:cpu (draw-points-host pos num-bodies))
    (:gpu (memcpy-array-device-to-host pos)
          (draw-points-host pos num-bodies))
    (:gpu/interop
          (draw-points-interop pos num-bodies))))

(defun draw-points-host (pos num-bodies)
  (gl:begin :points)
  (dotimes (i num-bodies)
    (multiple-value-bind (x y z _) (array-ref pos i)
      (declare (ignorable _))
      (gl:vertex x y z)))
  (gl:end))

(defun draw-points-interop (pos num-bodies)
  (declare (ignorable pos num-bodies))
  (not-implemented))


;;;
;;; Functions for texture creation
;;;

(defun eval-hermite (pa pb va vb u)
  (let ((u2 (* u u))
        (u3 (* u u u)))
    (let ((b0 (+ (- (* 2 u3) (* 3 u2)) 1))
          (b1 (+ (* -2 u3) (* 3 u2)))
          (b2 (+ (- u3 (* 2 u2)) u))
          (b3 (- u3 u)))
      (+ (* b0 pa) (* b1 pb) (* b2 va) (* b3 vb)))))

(defun create-gaussian-map (b n)
  (let ((incr (/ 2.0 n))
        (j 0))
    (do ((y 0 (1+ y))
         (yy -1.0 (+ yy incr)))
        ((= y n))
      (do ((x 0 (1+ x))
           (xx -1.0 (+ xx incr)))
          ((= x n))
        (let* ((dist (min (sqrt (+ (* xx xx) (* yy yy))) 1.0))
               (hermite (eval-hermite 1.0 0 0 0 dist))
               (value (truncate (* hermite 255))))
          (setf (cffi:mem-aref b :unsigned-char j)       value
                (cffi:mem-aref b :unsigned-char (+ j 1)) value
                (cffi:mem-aref b :unsigned-char (+ j 2)) value
                (cffi:mem-aref b :unsigned-char (+ j 3)) value)
          (incf j 4))))))


;;;
;;; Functions for initialize body position
;;;

(defun divided-point (inner outer k)
  (+ inner (* (- outer inner) k)))

(defun norm-float3 (x)
  (assert (float3-p x))
  (sqrt (+ (* (float3-x x) (float3-x x))
           (* (float3-y x) (float3-y x))
           (* (float3-z x) (float3-z x)))))

(defun normalize-float3 (x)
  (assert (float3-p x))
  (let ((r (norm-float3 x)))
    (if (< 1.0e-6 r)
        (make-float3 (/ (float3-x x) r)
                     (/ (float3-y x) r)
                     (/ (float3-z x) r))
        x)))

(defun cross (v0 v1)
  (assert (and (float3-p v0) (float3-p v1)))
  (make-float3 (- (* (float3-y v0) (float3-z v1))
                  (* (float3-z v0) (float3-y v1)))
               (- (* (float3-z v0) (float3-x v1))
                  (* (float3-x v0) (float3-z v1)))
               (- (* (float3-x v0) (float3-y v1))
                  (* (float3-y v0) (float3-x v1)))))

(defun randomize-bodies (pos vel cluster-scale velocity-scale num-bodies)
  (let* ((scale cluster-scale)
         (vscale (* scale velocity-scale))
         (inner (* 2.5 scale))
         (outer (* 4.0 scale)))
    (dotimes (i num-bodies)
      (let ((point (normalize-float3 (make-float3 (- (random 1.0) 0.5)
                                                  (- (random 1.0) 0.5)
                                                  (- (random 1.0) 0.5))))
            (k (divided-point inner outer (random 1.0))))
        (setf (array-ref pos i) (list (* (float3-x point) k)
                                      (* (float3-y point) k)
                                      (* (float3-z point) k)
                                      1.0))

      (let* ((axis (make-float3 0.0 0.0 1.0))
             (vv (multiple-value-bind (x y z _) (array-ref pos i)
                   (declare (ignorable _))
                   (cross (make-float3 x y z) axis))))
        (setf (array-ref vel i) (list (* (float3-x vv) vscale)
                                      (* (float3-y vv) vscale)
                                      (* (float3-z vv) vscale)
                                      1.0)))))))


;;;
;;; main
;;;

#|
(require :sb-sprof)
(defun main/profile ()
  (sb-sprof:with-profiling (:max-samples 1000
                            :report      :graph
                            :loop        nil)
    (setf *gpu* nil)
    (main)))
|#

(defun main ()
  (setf glut:*run-main-loop-after-display* nil)
  (let ((window (make-instance 'nbody-window)))
    (glut:display-window window) ; GLUT window must be created before initializing nbody-demo
    (with-nbody-demo (demo 2048 *gpu*)
      (with-frame-rate-counter (counter)
        (setf (slot-value window 'nbody-demo) demo
              (slot-value window 'counter)    counter)
        (glut:main-loop)))))

(defun not-implemented ()
  (error "not implemented."))
