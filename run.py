import cv2
import time

from utils.TFLiteFaceAlignment import * 
from utils.TFLiteFaceDetector import * 
from utils.functions import *

path = "./"

fd = UltraLightFaceDetecion(path + "utils/weights/RFB-320.tflite", conf_threshold=0.98)
fa = CoordinateAlignmentModel(path + "utils/weights/coor_2d106.tflite")

extend_pixel = 50

def getPose():
    # Open the webcam stream
    webcam_0 = cv2.VideoCapture(0)
    # if not webcam_0.isOpened():
    #     return True
    prev_frame_time = 0
    new_frame_time = 0
    queue = []
    count = 0

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_0.read()
        orig_image = cv2.flip(orig_image, 1)

        final_frame = orig_image.copy()

        temp_boxes, _ = fd.inference(orig_image)

        # Draw boundary boxes around faces
        draw_box(final_frame, temp_boxes, color=(125, 255, 125))

        # Find landmarks of each face
        temp_resized_marks = fa.get_landmarks(orig_image, temp_boxes)


        # Draw landmarks of each face
        for bbox_I, landmark_I in zip(temp_boxes, temp_resized_marks):
            roll, pitch, yaw, _ = estimatePose(final_frame, landmark_I)
            cv2.putText(final_frame, 'Roll: {0}'.format(roll), (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(final_frame, 'Pitch: {0}'.format(pitch), (20, 130), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(final_frame, 'Yaw: {0}'.format(yaw), (20, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)

            draw_landmark(final_frame, landmark_I, color=(125, 255, 125))
            # Show rotated face image
            # xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
            # xmin -= extend_pixel
            # xmax += extend_pixel
            # ymin -= 2 * extend_pixel
            # ymax += extend_pixel

            # xmin = 0 if xmin < 0 else xmin
            # ymin = 0 if ymin < 0 else ymin
            # xmax = frame_width if xmax >= frame_width else xmax
            # ymax = frame_height if ymax >= frame_height else ymax

            # face_I = orig_image[ymin:ymax, xmin:xmax]
            # face_I = align_face(face_I, landmark_I[34], landmark_I[88])

            # cv2.imshow('Rotated raw face image', face_I)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('', final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def demo():
    # Read a frame from the stream
    orig_image = cv2.imread('img/1.JPG')
    orig_image = cv2.resize(orig_image, (600, 400), interpolation = cv2.INTER_CUBIC)
    final_frame = orig_image.copy()

    temp_boxes, _ = fd.inference(orig_image)

    # Find landmarks of each face
    temp_resized_marks = fa.get_landmarks(orig_image, temp_boxes)

    for bbox_I, landmark_I in zip(temp_boxes, temp_resized_marks):
        roll, pitch, yaw, _ = estimatePose(final_frame, landmark_I)

        print("roll:", roll, " pitch:", pitch, "yaw:", yaw)

if __name__ == '__main__':
    # getPose()
    demo()