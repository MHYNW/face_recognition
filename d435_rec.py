import dlib, cv2
import numpy as np
import pyrealsense2 as rs

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

descs = np.load('faces/descs.npy')[()]

def encode_face(img):
  dets = detector(img, 1)

  if len(dets) == 0:
    return np.empty(0)

  for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    return np.array(face_descriptor)



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        #images = np.hstack((color_image, depth_colormap))

        dets = detector(color_image, 1)

        for k, d in enumerate(dets):
            shape = sp(color_image, d)
            face_descriptor = facerec.compute_face_descriptor(color_image, shape)

            last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0, 0, 255)}

            for name, saved_desc in descs.items():
                dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

                if dist < last_found['dist']:
                    last_found = {'name': name, 'dist': dist, 'color': (255, 255, 255)}

            cv2.rectangle(color_image, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'],
                          thickness=2)
            cv2.putText(color_image, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=last_found['color'], thickness=2)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streamingjh.jpg
    pipeline.stop()