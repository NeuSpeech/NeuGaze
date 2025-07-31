# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

import os

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def get_landmark_from_result(result,image_width=None,image_height=None):
    result = result.face_landmarks
    assert len(result)==1
    face_landmarks=result[0]
    face_landmarks_numpy=[]
    for i,landmark in enumerate(face_landmarks):
        x=landmark.x
        y=landmark.y
        z=landmark.z
        face_landmarks_numpy.append([x,y,z])
    face_landmarks_numpy=np.array(face_landmarks_numpy)
    return face_landmarks_numpy




def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    drawed_annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=drawed_annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=drawed_annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=drawed_annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return drawed_annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # indexes=[i for i,a in enumerate(face_blendshapes_names) if not a.startswith('eye')]
    # face_blendshapes_names=[a for i,a in enumerate(face_blendshapes_names) if i in indexes]
    # face_blendshapes_scores=[a for i,a in enumerate(face_blendshapes_scores) if i in indexes]
    # print(face_blendshapes_names)
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    plt.ylim()
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlim(0,1)
    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Convert RGB to BGR (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    plt.close(fig)

    return img

def real_time_plot_face_blendshapes_bar_graph(face_blendshapes,ax,fig):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))
    ax.clear()
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        ax.set_text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    fig.canvas.draw()

    fig.canvas.flush_events()
    plt.pause(0.02)

def print_result_func(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('face landmarker result: {}'.format(result))
    global annotated_image
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    # print(f'print_result_func annotated_image:{annotated_image}')


def crop_center_rectangle(image, rectangle_width, rectangle_height):
    # 获取图像的尺寸
    image_height, image_width = image.shape[:2]

    # 计算长方形在图像中的位置
    x = (image_width - rectangle_width) // 2
    y = (image_height - rectangle_height) // 2

    # 截取长方形
    rectangle = image[y:y + rectangle_height, x:x + rectangle_width]

    return rectangle


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080 )
    supported_modes=['livestream','video','image']
    assert mode in supported_modes
    if mode =='livestream':
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM
        print_result=print_result_func
    elif mode =='video':
        running_mode=mp.tasks.vision.RunningMode.VIDEO
        print_result=None
    elif mode =='image':
        running_mode=mp.tasks.vision.RunningMode.IMAGE
        print_result=None
    else:
        raise NotImplementedError
    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker_v2_with_blendshapes.task',
                                      )

    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           running_mode=running_mode,
                                           result_callback=print_result,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)

    detector = vision.FaceLandmarker.create_from_options(options)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720 )
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*2
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    path=f'videos/mediapipe/{mode}_output.mp4'
    os.makedirs(os.path.dirname(path),exist_ok=True)
    out = cv2.VideoWriter(path, fourcc, 25.0, (frame_width, frame_height))
    print(f"默认帧尺寸: 宽度 = {frame_width}, 高度 = {frame_height}")

    global annotated_image
    annotated_image = None
    counter=0
    time_list=[]

    # 使用示例
    # plotter = FaceBlendshapesPlotter()
    # plotter.start_animation()
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    while True:
        t_start=time.time()
        ret, image = cap.read()
        milliseconds = int(time.time()*1000)
        # print(milliseconds)
        if not ret:
            print("无法读取摄像头帧")
            break
        # print(image.shape)
        # image = cv2.flip(image, 1, 0)
        # image = cv2.bilateralFilter(image, 5, 50, 50)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        if mode =='livestream':
            detector.detect_async(mp_image, milliseconds)
            # print(f'annotated_image:{annotated_image}')
        elif mode == 'video':
            time1=time.time()
            face_landmarker_result = detector.detect_for_video(mp_image, milliseconds)
            time2=time.time()
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result)
            time3=time.time()
            print(f'time1:{time1-t_start},time2:{time2-time1},time3:{time3-time2}')
        elif mode=='image':
            face_landmarker_result = detector.detect(mp_image)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result)
        else:
            raise NotImplementedError
        # print(face_landmarker_result)
        # if len(face_landmarker_result.face_blendshapes) > 0:
        #     # print('face_landmarker_result')
        #     result_img=plot_face_blendshapes_bar_graph(face_landmarker_result.face_blendshapes[0])
        #     cv2.imshow('result', result_img)
        #     print(face_landmarker_result.facial_transformation_matrixes)
        if annotated_image is not None:
            # out.write(annotated_image)
            cv2.imshow('摄像头', annotated_image)
        else:
            cv2.imshow('摄像头', image)
        # cv2.imshow('摄像头', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        counter+=1
        t_end=time.time()
        time_list.append(t_end-t_start)
        print(f'FPS:{1/(sum(time_list[-30:])/len(time_list[-30:]))}')
        # 加了mediapipe的帧率是26到30FPS间，有浮动
        # 没有操作的情况下，FPS达到30FPS左右，说明这个mediapipe其实挺快的。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mode='video'
    main()
