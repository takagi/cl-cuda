#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

#|
  This file is based on the CUDA SDK's "nbody" sample.
|#

(in-package :cl-user)
(defpackage cl-cuda-interop-examples.nbody
  (:use :cl
        :cl-cuda-interop)
  (:export :main))
(in-package :cl-cuda-interop-examples.nbody)


;;;
;;; Vec3 array
;;;

(deftype vec3-array () '(simple-array single-float (*)))

(defun make-vec3-array (n)
  (let ((size (* n 3)))
    (make-array (list size) :element-type 'single-float
                            :initial-element 0.0)))

(defun vec3-array-dimension (array)
  (/ (array-dimension array 0) 3))

(defun vec3-aref (array i elm)
  (ecase elm
    (:x (aref array (+ (* i 3) 0)))
    (:y (aref array (+ (* i 3) 1)))
    (:z (aref array (+ (* i 3) 2)))))

(defun (setf vec3-aref) (val array i elm)
  (ecase elm
    (:x (setf (aref array (+ (* i 3) 0)) val))
    (:y (setf (aref array (+ (* i 3) 1)) val))
    (:z (setf (aref array (+ (* i 3) 2)) val))))

(defmacro with-vec3-array-values (((x y z) array i) &body body)
  ;; for fast read
  `(multiple-value-bind (,x ,y ,z) (vec3-array-values ,array ,i)
     (declare (type single-float ,x ,y ,z))
     ,@body))

(declaim (inline vec3-array-values))
(defun vec3-array-values (array i)
  (declare (optimize (speed 3) (safety 0)))
  (declare (type vec3-array array)
           (type fixnum i))
  (values (aref array (the fixnum (+ (the fixnum (* i 3)) 0)))
          (aref array (the fixnum (+ (the fixnum (* i 3)) 1)))
          (aref array (the fixnum (+ (the fixnum (* i 3)) 2)))))

(defmacro set-vec3-array ((array i) x y z)
  ;; for fast write
  `(%set-vec3-array ,array ,i ,x ,y ,z))

(declaim (inline %set-vec3-array))
(defun %set-vec3-array (array i x y z)
  (declare (optimize (speed 3) (safety 0)))
  (declare (type vec3-array array)
           (type fixnum i))
  (setf (aref array (the fixnum (+ (the fixnum (* i 3)) 0))) x
        (aref array (the fixnum (+ (the fixnum (* i 3)) 1))) y
        (aref array (the fixnum (+ (the fixnum (* i 3)) 2))) z))


;;;
;;; Abstract array structure
;;;

(defun alloc-array (n gpu interop)
  (unless (not (and (null gpu) interop))
    (error "Interoperability is available on GPU only."))
  (cond
    ((and gpu interop)
     (list :gpu/interop (cl-cuda-interop:alloc-memory-block 'float4 n)))
    (gpu (list :gpu (cl-cuda:alloc-memory-block 'float4 n)))
    (t (list :cpu (make-vec3-array n)))))

(defun free-array (array)
  (ecase (array-type array)
    (:gpu/interop (cl-cuda-interop:free-memory-block (raw-array array)))
    (:gpu (cl-cuda:free-memory-block (raw-array array)))
    (:cpu nil)))

(defun array-ref (array i)
  (let ((type (array-type array))
        (raw-array (raw-array array)))
    (ecase type
      (:gpu/interop
       (let ((x (cl-cuda-interop:memory-block-aref raw-array i)))
         (values (float4-x x) (float4-y x) (float4-z x) (float4-w x))))
      (:gpu
       (let ((x (cl-cuda:memory-block-aref raw-array i)))
         (values (float4-x x) (float4-y x) (float4-z x) (float4-w x))))
      (:cpu
       (values (vec3-aref raw-array i :x)
               (vec3-aref raw-array i :y)
               (vec3-aref raw-array i :z)
               1.0)))))

(defun (setf array-ref) (val array i)
  (destructuring-bind (x y z w) val
    (let ((type (array-type array))
          (raw-array (raw-array array)))
      (ecase type
        (:gpu/interop (setf (cl-cuda-interop:memory-block-aref raw-array i)
                            (make-float4 x y z w)))
        (:gpu (setf (cl-cuda:memory-block-aref raw-array i)
                    (make-float4 x y z w)))
        (:cpu (setf (vec3-aref raw-array i :x) x
                    (vec3-aref raw-array i :y) y
                    (vec3-aref raw-array i :z) z))))))

(defun array-type (array)
  (car array))

(defun raw-array (array)
  (cadr array))

(defun sync-array (array direction)
  (let ((array-type (array-type array))
        (raw-array (raw-array array)))
    (ecase array-type
      (:gpu/interop (cl-cuda-interop:sync-memory-block raw-array direction))
      (:gpu (cl-cuda:sync-memory-block raw-array direction)))))


;;;
;;; Kernel functions
;;;

(defkernel body-body-interaction (float3 ((ai float3) (bi float4) (bj float4)))
  (let* ((r (float3 (- (float4-x bj) (float4-x bi))
                    (- (float4-y bj) (float4-y bi))
                    (- (float4-z bj) (float4-z bi))))
         (softening-squared (* 0.1 0.1))
         (dist-sqr (+ (* (float3-x r) (float3-x r))
                      (* (float3-y r) (float3-y r))
                      (* (float3-z r) (float3-z r))
                      softening-squared))
         (inv-dist (rsqrt dist-sqr))
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
    (let* ((acc (float3 0.0 0.0 0.0))
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
    (let* ((position (aref old-pos index))
           (accel (compute-body-accel position old-pos total-num-bodies))
;           (accel (compute-body-accel-without-shared-memory position old-pos total-num-bodies))
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
    (with-vec3-array-values ((x1 y1 z1) old-pos i)
      (dotimes (j total-num-bodies)
        (with-vec3-array-values ((x2 y2 z2) old-pos j)
          (when (/= i j)
            (multiple-value-bind (ax ay az) (body-body-interaction-cpu x1 y1 z1 x2 y2 z2)
              (declare (type single-float ax ay az))
              (let ((k (* delta-time damping)))
                (with-vec3-array-values ((vx vy vz) vel i)
                  (set-vec3-array (vel i) (+ vx (* ax k))
                                          (+ vy (* ay k))
                                          (+ vz (* az k)))))))))
      (with-vec3-array-values ((vx vy vz) vel i)
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
   (counter :initarg :counter))
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
    ;; measure and display frame rate
    (measure-framerate-counter counter)
    (let ((fps (framerate-counter-fps counter))
          (num-bodies (nbody-demo-num-bodies nbody-demo)))
      (display-framerate fps num-bodies))))

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

(defun compute-pref-stats (fps num-bodies interactions)
  (let* ((interactions-per-second
           (/ (* num-bodies num-bodies interactions)
              (/ 1.0 fps)
              1.0e9))
         (gflops (* interactions-per-second *flops-per-interaction*)))
    (values interactions-per-second gflops)))

(let ((previous 0.0))
  (defun display-framerate (fps num-bodies)
    (when (/= fps previous)
      (setf previous fps)
      (multiple-value-bind (interactions-per-second gflops)
          (compute-pref-stats fps num-bodies 1)
        (let ((msg (format nil "CUDA N-Body (~A bodies): ~,1F fps | ~,1F BIPS | ~,1F GFLOP/s | single precision~%"
                           num-bodies fps interactions-per-second gflops)))
          (format t msg)
          (glut:set-window-title msg))))))


;;;
;;; Framerate counter
;;; * note that since a framerate counter uses CUDA event, it depends on
;;;   CUDA context
;;;

(defstruct (framerate-counter (:constructor %make-framerate-counter))
  (timer :timer :read-only t)
  (fps-count 0)
  (fps-limit 5)
  (fps 0.0))

(defun init-framerate-counter ()
  (let ((timer (create-timer)))
    (start-timer timer)
    (%make-framerate-counter :timer timer)))

(defun release-framerate-counter (counter)
  (destroy-timer (framerate-counter-timer counter)))

(defmacro with-framerate-counter ((var) &body body)
  `(let ((,var (init-framerate-counter)))
     (unwind-protect (progn ,@body)
       (release-framerate-counter ,var))))

(defun measure-framerate-counter (counter)
  (symbol-macrolet ((timer (framerate-counter-timer counter))
                    (fps (framerate-counter-fps counter))
                    (fps-count (framerate-counter-fps-count counter))
                    (fps-limit (framerate-counter-fps-limit counter)))
    ;; increment fps-count
    (incf fps-count)
    ;; compute fps for certain period
    (when (>= fps-count fps-limit)
      (let* ((elapsed-time (progn
                             (stop-timer timer)
                             (synchronize-timer timer)
                             (prog1 (elapsed-time timer)
                               (start-timer timer))))
             (milliseconds (/ elapsed-time fps-count)))
        (setf fps (/ 1.0 (/ milliseconds 1000.0))))
      (setf fps-limit (max fps 1.0))
      (setf fps-count 0))))


;;;
;;; NBody Demo
;;;

(defstruct (nbody-demo (:constructor %make-nbody-demo))
  system renderer)

(defun init-nbody-demo (num-bodies &key (gpu t) (interop nil))
  (let ((system (init-body-system num-bodies :gpu gpu :interop interop))
        (renderer (init-particle-renderer)))
    ;; Reset body system
    (reset-body-system system)
    ;; Synchronize memory form host to device
    (when (body-system-gpu system)
      (sync-body-system system :host-to-device))
    ;; Make NBody demo
    (%make-nbody-demo :system system :renderer renderer)))

(defun release-nbody-demo (demo)
  (release-particle-renderer (nbody-demo-renderer demo))
  (release-body-system (nbody-demo-system demo)))

(defmacro with-nbody-demo ((var num-bodies &key (gpu t) (interop nil))
                           &body body)
  `(let ((,var (init-nbody-demo ,num-bodies :gpu ,gpu :interop ,interop)))
     (unwind-protect (progn ,@body)
       (release-nbody-demo ,var))))

(defun update-nbody-demo (demo)
  (update-body-system (nbody-demo-system demo)))

(defun display-nbody-demo (demo)
  (let ((system (nbody-demo-system demo))
        (renderer (nbody-demo-renderer demo)))
    (let ((pos (body-system-old-pos system))
          (num-bodies (body-system-num-bodies system)))
      (display-particle-renderer renderer pos num-bodies :particle-sprites))))

(defun nbody-demo-num-bodies (demo)
  (body-system-num-bodies (nbody-demo-system demo)))


;;;
;;; Body System
;;;

(defstruct (body-system (:constructor %make-body-system))
  ;; Memory blocks
  (new-pos :new-pos)                    ; Not read-only to flip
  (old-pos :old-pos)
  (vel :vel :read-only t)
  ;; Flags
  (gpu :gpu :read-only t)
  (interop :interop :read-only t)
  ;; Number of bodies
  (num-bodies :num-bodies :read-only t)
  ;; Simulation parameters
  (delta-time     0.016 :read-only t)
  (damping        1.0   :read-only t)
  (block-dim      256   :read-only t)
  (cluster-scale  1.56  :read-only t)
  (velocity-scale 2.64  :read-only t))

(defun init-body-system (num-bodies &key (gpu t) (interop nil))
  (unless (not (and (null gpu) interop))
    (error "Interoperability is available on GPU only."))
  ;; Allocate array
  (let ((new-pos (alloc-array num-bodies gpu interop))
        (old-pos (alloc-array num-bodies gpu interop))
        (vel (alloc-array num-bodies gpu interop)))
    (%make-body-system :new-pos new-pos
                       :old-pos old-pos
                       :vel vel
                       :gpu gpu
                       :interop interop
                       :num-bodies num-bodies)))

(defun release-body-system (system)
  (free-array (body-system-new-pos system))
  (free-array (body-system-old-pos system))
  (free-array (body-system-vel system)))

(defun reset-body-system (system)
  (let ((old-pos (body-system-old-pos system))
        (vel (body-system-vel system))
        (cluster-scale (body-system-cluster-scale system))
        (velocity-scale (body-system-velocity-scale system))
        (num-bodies (body-system-num-bodies system)))
    (randomize-bodies old-pos vel
                      cluster-scale velocity-scale num-bodies)))

(defun sync-body-system (system direction)
  (sync-array (body-system-new-pos system) direction)
  (sync-array (body-system-old-pos system) direction)
  (sync-array (body-system-vel system) direction))

(defun update-body-system (system)
  ;; Integrate NBody system
  (let ((new-pos (body-system-new-pos system))
        (old-pos (body-system-old-pos system))
        (vel (body-system-vel system))
        (delta-time (body-system-delta-time system))
        (damping (body-system-damping system))
        (num-bodies (body-system-num-bodies system))
        (p (body-system-block-dim system)))
    (if (body-system-gpu system)
        (integrate-nbody-system-gpu new-pos old-pos vel
                                    delta-time damping num-bodies p)
        (integrate-nbody-system-cpu new-pos old-pos vel
                                    delta-time damping num-bodies)))
  ;; Flip position arrays
  (symbol-macrolet ((new-pos (body-system-new-pos system))
                    (old-pos (body-system-old-pos system)))
    (rotatef new-pos old-pos)))


;;;
;;; Particle renderer
;;;

(defparameter +vertex-shader+
  (cl-cuda.lang.util:unlines
    "void main()                                                            "
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
  (cl-cuda.lang.util:unlines
     "uniform sampler2D splatTexture;                                        "
     "void main()                                                            "
     "{                                                                      "
     "    vec4 color2 = gl_SecondaryColor;                                   "
     "    vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st);"
     "    gl_FragColor =                                                     "
     "         color * color2;" ;mix(vec4(0.1, 0.0, 0.0, color.w), color2, color.w);"
     "}                                                                      "))

(defstruct (particle-renderer (:constructor %make-particle-renderer))
  ;; for rendering
  (point-size  1.0                :read-only t)
  (sprite-size 2.0                :read-only t)
  (base-color  #(1.0 0.6 0.3 1.0) :read-only t)
  ;; for shader
  program vertex-shader pixel-shader
  ;; for texture
  texture texture-data)

(defun init-particle-renderer ()
  (let ((renderer (%make-particle-renderer)))
    (init-shader renderer)
    (create-texture renderer 32)
    renderer))

(defun init-shader (renderer)
  (symbol-macrolet
      ((program (particle-renderer-program renderer))
       (vertex-shader (particle-renderer-vertex-shader renderer))
       (pixel-shader (particle-renderer-pixel-shader renderer)))
    ;; create shader objects
    (setf vertex-shader (gl:create-shader :vertex-shader)
          pixel-shader (gl:create-shader :fragment-shader))
    ;; set shader source codes
    (gl:shader-source vertex-shader +vertex-shader+)
    (gl:shader-source pixel-shader +pixel-shader+)
    ;; compile shader source codes
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
  (symbol-macrolet ((texture-data (particle-renderer-texture-data renderer))
                    (texture (particle-renderer-texture renderer)))
    ;; prepare texture data
    (setf texture-data (cffi:foreign-alloc :unsigned-char
                                           :count (* 4 resolution
                                                       resolution)))
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
  (release-shader renderer))

(defun release-shader (renderer)
  (declare (ignorable renderer))
  (values))

(defun destroy-texture (renderer)
  (cffi:foreign-free (particle-renderer-texture-data renderer)))

(defun display-particle-renderer (renderer pos num-bodies mode)
  (let ((point-size  (particle-renderer-point-size  renderer))
        (sprite-size (particle-renderer-sprite-size renderer))
        (program     (particle-renderer-program     renderer))
        (texture     (particle-renderer-texture     renderer))
        (base-color  (particle-renderer-base-color  renderer)))
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
    (:gpu (sync-array pos :device-to-host)
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
  (let ((pbo (memory-block-vertex-buffer-object (raw-array pos))))
    ;; enable GL_VERTEX_ARRAY
    (gl:enable-client-state :vertex-array)
    ;; bind a named buffer object
    (gl:bind-buffer :array-buffer pbo)
    ;; define an array of vertex data
    (%gl:vertex-pointer 4 :float 0 (cffi:null-pointer))
    ;; render primitives from array data
    (gl:draw-arrays :points 0 num-bodies)
    ;; unbind a buffer object
    (gl:bind-buffer :array-buffer 0)
    ;; disable GL_VERTEX_ARRAY
    (gl:disable-client-state :vertex-array)))


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
(eval-when (:compile-toplevel :load-toplevel)
  (require :sb-sprof))
(defun main/profile (&key (gpu t) (interop nil))
  (sb-sprof:with-profiling (:max-samples 1000
                            :report      :graph
                            :loop        nil)
    (main :gpu gpu :interop interop)))
|#

(defun main (&key (gpu t) (interop nil))
  (let ((dev-id 0)
        (glut:*run-main-loop-after-display* nil)
        (window (make-instance 'nbody-window)))
    (glut:display-window window) ; GLUT window must be created before initializing CUDA
    (let ((cl-cuda-interop:*show-messages* nil))
      (with-cuda (dev-id :interop interop)
        (with-framerate-counter (counter)
          (with-nbody-demo (demo 2048 :gpu gpu :interop interop)
            (setf (slot-value window 'nbody-demo) demo
                  (slot-value window 'counter) counter)
            (glut:main-loop)))))))
