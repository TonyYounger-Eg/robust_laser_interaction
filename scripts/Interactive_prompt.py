# coding:utf-8
#
# Interactive_prompt.py
#
#  Created on: 2024/4/7
#      Author: Tex Yan Liu
#
# description: Ros node for the module of interactive_prompt

import argparse  # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import os
import shutil
import rospy
import time
import cv2  # 惊天大BUG，cv2必须在torch上面，否则会RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
import torch
import torch.backends.cudnn as cudnn  # cuda模块

from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import quaternion
from geometry_msgs.msg import PoseStamped
import rospy

import numpy as np
import pyrealsense2 as rs  # 导入realsense的sdk模块
import threading
import open3d as o3d
from pyinstrument import Profiler
import copy

from Anything_Grasping.msg import prompt_info
from Anything_Grasping.msg import grasp_info
from segment_anything import sam_model_registry, SamPredictor  # 被点图片识别
from grasp_wrench_2d_v6 import grasp_wrench_2d
from std_msgs.msg import String

from gmm import gmm

rospy.init_node('interactive_prompt', anonymous=True)
prompt_pub = rospy.Publisher('auto_grasp/interactive_prompt', prompt_info, queue_size=1)
grasp_pub = rospy.Publisher('auto_grasp/grasp_execution', grasp_info, queue_size=1)
laser_pub = rospy.Publisher('auto_grasp/laser_waiter', String, queue_size=1)
grasp_info = grasp_info()
prompt_info = prompt_info()

laser_callback_status = 'laser_back_is_waiting'
pose_adjustment_status = ''
grasp_status_info = ''

ARUCO_DICT = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
              "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
              "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
              "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
              "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
              "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
              "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
              "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
              "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
              #	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
              #	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
              }

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_ARUCO_ORIGINAL"])
arucoParams = cv2.aruco.DetectorParameters()
matrix_coefficients_1 = np.array([[603.74, 0.0, 311.225], [0.0, 602.553, 255.996], [0.0, 0.0, 1.0]])
distortion_coefficients_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

resolution_x = 640
resolution_y = 480

jaco_position = np.ones(3)
jaco_orientation = np.ones(4)

try:
    # 定义realsense相机的API
    pipeline_1 = rs.pipeline()  # 创建管道-这是流媒体和处理帧的顶级API 该管道简化了用户与设备和计算机视觉处理模块的交互。
    config_1 = rs.config()  # 该配置允许管道用户为管道流以及设备选择和配置请求过滤器。
    config_1.enable_stream(rs.stream.depth, resolution_x, resolution_y, rs.format.z16, 30)  # 配置深度图像
    config_1.enable_stream(rs.stream.color, resolution_x, resolution_y, rs.format.bgr8, 30)  # 配置彩色图像
    config_1.enable_device(str(134222072210))    # 1号相机的编号 134222072210
    pipeline_wrapper_1 = rs.pipeline_wrapper(pipeline_1)  # 管道握手函数
    pipeline_profile_1 = config_1.resolve(pipeline_wrapper_1)  # 管道配置
    pipeline_1.start(config_1)  # 启动相关配置的API
    rgbd_device_1 = pipeline_profile_1.get_device()  # 获取设备
    eyetohand_camera_status = True
    # print(rgbd_device_1)
    found_rgb_1 = False
    print("Eye_to_hand_realsense_D435_status_is ok")

    for s_1 in rgbd_device_1.sensors:  # 验证相机信息是否拉取完全
        if s_1.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb_1 = True
            break
    if not found_rgb_1:
        print("The Manual_demo requires Depth camera with Color sensor")
        exit(0)
except RuntimeError:
    eyetohand_camera_status = False
    print("Eye_to_hand_realsense_D435_RuntimeError")

try:
    # 定义realsense相机的API
    pipeline_2 = rs.pipeline()  # 创建管道-这是流媒体和处理帧的顶级API 该管道简化了用户与设备和计算机视觉处理模块的交互。
    config_2 = rs.config()  # 该配置允许管道用户为管道流以及设备选择和配置请求过滤器。
    config_2.enable_stream(rs.stream.depth, resolution_x, resolution_y, rs.format.z16, 30)  # 配置深度图像
    config_2.enable_stream(rs.stream.color, resolution_x, resolution_y, rs.format.bgr8, 30)  # 配置彩色图像
    config_2.enable_device(str(145422071093))    # 145422071093 二号相机的编号
    pipeline_wrapper_2 = rs.pipeline_wrapper(pipeline_2)  # 管道握手函数
    pipeline_profile_2 = config_2.resolve(pipeline_wrapper_2)  # 管道配置
    pipeline_2.start(config_2)  # 启动相关配置的API
    rgbd_device_2 = pipeline_profile_2.get_device()  # 获取设备
    # print(rgbd_device_2)
    found_rgb_2 = False
    eyeonhand_camera_status = True
    print("Eye_on_hand_realsense_D435_status_is ok")
    for s_2 in rgbd_device_2.sensors:  # 验证相机信息是否拉取完全
        if s_2.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb_2 = True
            break
    if not found_rgb_2:
        print("The Manual_demo requires Depth camera with Color sensor")
        exit(0)
except RuntimeError:
    eyeonhand_camera_status = False
    print("Eye_on_hand_realsense_D435_RuntimeError")


# realsense图像对齐函数
def get_aligned_images_1():
    # 创建对齐对象与color流对齐
    align_to_1 = rs.stream.color
    align_1 = rs.align(align_to_1)

    frames_1 = pipeline_1.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames_1 = align_1.process(frames_1)  # 获取对齐帧，将深度框与颜色框对齐

    d435_aligned_depth_frame_1 = aligned_frames_1.get_depth_frame()  # 获取对齐帧中的的depth帧
    d435_aligned_color_frame_1 = aligned_frames_1.get_color_frame()  # 获取对齐帧中的的color帧

    # 将images转为numpy arrays
    d435_img_color_1 = np.asanyarray(d435_aligned_color_frame_1.get_data())  # BGR图
    d435_img_depth_1 = np.asanyarray(d435_aligned_depth_frame_1.get_data())  # 深度图

    # 获取相机参数
    d435_depth_intrin_1 = d435_aligned_depth_frame_1.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    d435_color_intrin_1 = d435_aligned_color_frame_1.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    # print("深度内参1", d435_depth_intrin_1)
    # print("彩色内参", d435_color_intrin_1)  # 目前看来，他和深度内参的一样
    d435_depth_to_color_extrin_1 = d435_aligned_depth_frame_1.profile.get_extrinsics_to(d435_aligned_color_frame_1.profile)
    d435_color_to_depth_extrin_1 = d435_aligned_color_frame_1.profile.get_extrinsics_to(d435_aligned_depth_frame_1.profile)

    d435_depth_mapped_image_1 = cv2.applyColorMap(cv2.convertScaleAbs(d435_img_depth_1, alpha=0.03), cv2.COLORMAP_JET)

    # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
    return d435_color_intrin_1, d435_depth_intrin_1, d435_img_color_1, d435_img_depth_1, d435_depth_mapped_image_1, \
        d435_aligned_color_frame_1, d435_aligned_depth_frame_1, d435_depth_to_color_extrin_1, d435_color_to_depth_extrin_1


# realsense图像对齐函数
def get_aligned_images_2():
    # 创建对齐对象与color流对齐
    align_to_2 = rs.stream.color
    align_2 = rs.align(align_to_2)

    frames_2 = pipeline_2.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames_2 = align_2.process(frames_2)  # 获取对齐帧，将深度框与颜色框对齐

    d435_aligned_depth_frame_2 = aligned_frames_2.get_depth_frame()  # 获取对齐帧中的的depth帧
    d435_aligned_color_frame_2 = aligned_frames_2.get_color_frame()  # 获取对齐帧中的的color帧

    # 将images转为numpy arrays
    d435_img_color_2 = np.asanyarray(d435_aligned_color_frame_2.get_data())  # BGR图
    d435_img_depth_2 = np.asanyarray(d435_aligned_depth_frame_2.get_data())  # 深度图

    # 获取相机参数
    d435_depth_intrin_2 = d435_aligned_depth_frame_2.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    d435_color_intrin_2 = d435_aligned_color_frame_2.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    # print("深度内参2", d435_depth_intrin_2)
    # print("彩色内参", d435_color_intrin)  # 目前看来，他和深度内参的一样
    d435_depth_to_color_extrin_2 = d435_aligned_depth_frame_2.profile.get_extrinsics_to(d435_aligned_color_frame_2.profile)
    d435_color_to_depth_extrin_2 = d435_aligned_color_frame_2.profile.get_extrinsics_to(d435_aligned_depth_frame_2.profile)

    d435_depth_mapped_image_2 = cv2.applyColorMap(cv2.convertScaleAbs(d435_img_depth_2, alpha=0.03), cv2.COLORMAP_JET)

    # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
    return d435_color_intrin_2, d435_depth_intrin_2, d435_img_color_2, d435_img_depth_2, d435_depth_mapped_image_2, \
        d435_aligned_color_frame_2, d435_aligned_depth_frame_2, d435_depth_to_color_extrin_2, d435_color_to_depth_extrin_2


def yolov8_detect(img_color_2):
    """
    YOLO
    """
    results = model.predict(img_color_2)
    annotated_frame = results[0].plot(conf=False, probs=False)
    laser_boxes = results[0].boxes.xywh.int().cpu().tolist()
    # print("函数中的laser_boxes:", laser_boxes)

    return annotated_frame, laser_boxes


def yolov8_track(img_color_2):
    """
    YOLO
    """
    # results = model.track(img_color_2, persist=True, conf=0.2, iou=0.5, tracker="bytetrack.yaml")  # 使用默认追踪器进行追踪
    results = model.track(img_color_2, persist=True, conf=0.2, iou=0.5, tracker="bytetrack.yaml")  # 使用默认追踪器进行追踪
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    laser_boxes = results[0].boxes.xywh.int().cpu().tolist()
    # print("函数中的laser_boxes:", laser_boxes)
    if results[0].boxes.id != None:
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    # print("laser_boxes:", laser_boxes)
    return annotated_frame, laser_boxes


def Aruco_marker(img_color_1):
    """
    Aruco_marker
    """
    ArucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = ArucoDetector.detectMarkers(img_color_1)
    # print("corners", corners)
    # print("ids", ids.shape)
    # print("rejected", rejected)
    # verify *at least* one ArUco marker was detected

    ids_set = []
    corners_set = []
    if ids is not None:
        for j in range(0, len(ids)):
            if ids[j][0] == 88:
                ids_set.append([88])
                corners_set.append(corners[j])

    # print("ids_set", ids_set)
    # print("标定板的像素角点", corners_set)
    matrix_set = []
    if len(corners_set) > 0:
        for i in range(0, len(ids_set)):
            ObjectP = np.zeros((4, 3))
            ml = 0.03829   # 129.49mm竖，128.9横着  六面体激光笔0.0380 0.3805都行  三面体激光笔0.03829
            ObjectP[0] = np.array([-ml / 2, ml / 2, 0])
            ObjectP[1] = np.array([ml / 2, ml / 2, 0])
            ObjectP[2] = np.array([ml / 2, -ml / 2, 0])
            ObjectP[3] = np.array([-ml / 2, -ml / 2, 0])
            _, rvec, tvec = cv2.solvePnP(ObjectP, corners_set[i], matrix_coefficients_1, distortion_coefficients_1)
            # print("rvec", rvec)
            cv2.aruco.drawDetectedMarkers(img_color_1, np.array(corners_set), np.array(ids_set))
            cv2.drawFrameAxes(img_color_1, matrix_coefficients_1, distortion_coefficients_1, rvec, tvec, 0.08,
                              thickness=1)   # 蓝色是z轴 绿色是y轴 红色是x轴
            """
            cv2.Rodrigues函数有问题，从旋转向量计算出来的矩阵不准
            20240601从点云看很准， Rxyz_to_Quaternion反倒是不准了。但是相机标定的时候，Rxyz_to_Quaternion是准的
            """
            rotation_matrix, jacobian_matrix = cv2.Rodrigues(rvec)
            # quaternion_from_rvec = Rxyz_to_Quaternion(rvec)
            # rotation_matrix = quaternion.as_rotation_matrix(quaternion_from_rvec)
            marker_matrix = np.eye(4)
            marker_matrix[:3, :3] = rotation_matrix
            marker_matrix[:3, 3] = tvec.reshape((1, 3))
            laser_pointer_matrix = np.eye(4)
            laser_pointer_matrix[:3, 3] = np.array([0, 0, -0.01814])     # 6面体标定板的尺寸0.05555，3面体标定板的尺寸0.01814，指环标定板尺寸0.018
            # laser_pointer_matrix[:3, 3] = np.array([0, 0, 0])   # 填为0意味着不改变其矩阵
            # print("0.标定板本身的姿态矩阵:\n", marker_matrix)
            # print("0.5 激光笔相对标定板的姿态矩阵:\n", laser_pointer_matrix)
            laser_eystohand_camera_matrix = np.matmul(marker_matrix, laser_pointer_matrix)

            # print("marker_matrix", marker_matrix)
            matrix_set.append(laser_eystohand_camera_matrix)
    # print("len(matrix_set)", len(matrix_set))
    # if len(matrix_set) > 1:
    #     print("Difference", matrix_set[0]-matrix_set[1])

    return img_color_1, corners_set, np.array(matrix_set)


laser_coor_1 = np.zeros(3)
laser_coor_2 = np.zeros(3)
laser_coor_3 = np.zeros(3)
ax = plt.axes(projection='3d')
plt.ion()  # interactive mode on!!!! 很重要，有了他就不需要plt.show()了


def draw_dynamic(matrix_set, index):
    global laser_coor_1, laser_coor_2, laser_coor_3
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # scale = 1
    # ax.set_xlim(-scale, scale)
    # ax.set_ylim(-scale, scale)
    # ax.set_zlim(-scale, scale)

    ax.set_title('3d_mobile_obs')
    plt.grid(True)
    point = []
    for k in range(len(matrix_set)):
        point.append(matrix_set[k][:3, 3])

    if index == 0:
        laser_coor_1 = point[0]
        laser_coor_2 = point[1]
        # laser_coor_3 = point[2]
    else:
        laser_coor_1 = np.append(laser_coor_1, point[0])
        laser_coor_1 = laser_coor_1.reshape(-1, 3)
        laser_coor_2 = np.append(laser_coor_2, point[1])
        laser_coor_2 = laser_coor_2.reshape(-1, 3)
        # laser_coor_3 = np.append(laser_coor_3, point[2])
        # laser_coor_3 = laser_coor_2.reshape(-1, 3)

    ax.scatter3D(laser_coor_1.T[0], laser_coor_1.T[1], laser_coor_1.T[2], c="blue")
    ax.scatter3D(laser_coor_2.T[0], laser_coor_2.T[1], laser_coor_2.T[2], c="red")
    # ax.scatter3D(laser_coor_3.T[0], laser_coor_3.T[1], laser_coor_3.T[2], c="green")
    plt.pause(0.001)


def Rxyz_to_Quaternion(vector): #输入的Rx, Ry, Rz为角度

    X, Y, Z = vector[0], vector[1], vector[2]
    # print("X", X)
    # print("Y", Y)
    # print("Z", Z)

    Qx = math.cos(Y/2)*math.cos(Z/2)*math.sin(X/2)-math.sin(Y/2)*math.sin(Z/2)*math.cos(X/2)
    Qy = math.sin(Y/2)*math.cos(Z/2)*math.cos(X/2)+math.cos(Y/2)*math.sin(Z/2)*math.sin(X/2)
    Qz = math.cos(Y/2)*math.sin(Z/2)*math.cos(X/2)-math.sin(Y/2)*math.cos(Z/2)*math.sin(X/2)
    Qw = math.cos(Y/2)*math.cos(Z/2)*math.cos(X/2)+math.sin(Y/2)*math.sin(Z/2)*math.sin(X/2)
    return np.quaternion(Qw, Qx, Qy, Qz)


def quaternion_to_rotation_matrix(quaternion):  #

    q0, q1, q2, q3 = quaternion.w,  quaternion.x, quaternion.y, quaternion.z   # q0->qw q1->qx q2->qy q3->qz
    rotation_matrix = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])
    return rotation_matrix


def call_kinova_pose():

    kinova_data = rospy.wait_for_message("/j2n6s300_driver/out/tool_pose", PoseStamped, timeout=None)
    # Jaco矩阵
    # 机械臂当前回调的姿态和坐标
    jaco_position = np.array([kinova_data.pose.position.x,
                              kinova_data.pose.position.y,
                              kinova_data.pose.position.z])
    jaco_orientation = np.quaternion(kinova_data.pose.orientation.w,
                                     kinova_data.pose.orientation.x,
                                     kinova_data.pose.orientation.y,
                                     kinova_data.pose.orientation.z)
    jaco_r_matrix = quaternion.as_rotation_matrix(jaco_orientation)
    jaco_matrix = np.eye(4)
    jaco_matrix[:3, :3] = jaco_r_matrix
    jaco_matrix[:3, 3] = jaco_position
    # print("机械臂自身姿态矩阵为:", jaco_matrix)
    return jaco_matrix


def get_3d_camera_coordinate(get_coord_tmp_depth_pixel, get_coord_tmp_aligned_depth_frame, get_coord_tmp_depth_intrin):
    x = int(get_coord_tmp_depth_pixel[0])
    y = int(get_coord_tmp_depth_pixel[1])
    get_coord_tmp_distance = get_coord_tmp_aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    get_coord_tmp_camera_coordinate = rs.rs2_deproject_pixel_to_point(get_coord_tmp_depth_intrin,
                                                                      get_coord_tmp_depth_pixel,
                                                                      get_coord_tmp_distance)
    # print ('camera_coordinate: ',camera_coordinate)
    get_coord_tmp_camera_coordinate_round = [round(num, 5) for num in get_coord_tmp_camera_coordinate]
    return get_coord_tmp_distance, get_coord_tmp_camera_coordinate_round


def angle_between_vectors(a, b):
    # 计算点积
    dot_product = np.dot(a, b)
    # 计算范数
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # 计算余弦值
    cos_theta = dot_product / (norm_a * norm_b)
    # 为了防止由于浮点数精度导致的cos_theta超过[-1, 1]范围
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # 计算夹角
    radian = np.arccos(cos_theta)
    angle = np.degrees(radian)
    # 返回角度（弧度制）
    return angle


def get_pointcloud(pointcloud_tmp_color_rs, pointcloud_tmp_depth_rs, index):  # 对齐后的彩色图和深度图作为输入，不是彩色帧的数组和深度帧的数组
    # 因为open3d处理的是RGB的，而realsense出来的是BGR的，需要在此处转换以下颜色通道
    pointcloud_tmp_color_rs = cv2.cvtColor(pointcloud_tmp_color_rs, cv2.COLOR_BGR2RGB)
    pointcloud_tmp_color_rs = o3d.geometry.Image(pointcloud_tmp_color_rs)
    pointcloud_tmp_depth_rs = o3d.geometry.Image(pointcloud_tmp_depth_rs)

    pointcloud_tmp_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(pointcloud_tmp_color_rs,
                                                                                   pointcloud_tmp_depth_rs,
                                                                                   depth_scale=1000.0,
                                                                                   depth_trunc=3.0,
                                                                                   convert_rgb_to_intensity=False)

    if index == 2:
        open3d_process_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            pointcloud_tmp_rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                640, 480, 605.471, 604.936, 327.157, 247.894))  # 这个地方默认是识别不了的，需要改成相机D435的内参
    else:
        open3d_process_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            pointcloud_tmp_rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                640, 480, 603.74, 602.553, 311.225, 255.996))  # 这个地方默认是识别不了的，需要改成相机D435的内参
    open3d_process_pointcloud.estimate_normals()

    passthrough_bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, 0),
                                                                   max_bound=(1, 1, 1.5))  # 直通滤波全局点云  # 蓝色z 红x 绿y
    pcl_fil = open3d_process_pointcloud.crop(passthrough_bounding_box)

    return pcl_fil


sam_checkpoint = "../segment-anything/checkpoints/sam_vit_h_4b8939.pth"  # sam的权重文件
sam_model_type = "vit_h"  # 模型类型
sam_device = "cuda"  # 应用 cuda
print("launch processing segment")
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam.to(device=sam_device)
sam_predictor = SamPredictor(sam)


# 定义一个MyThread.py线程类，构造多线程，用于SAM计算分割
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.result = None
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


def SAM_wrapper(frame, seg_input_point, seg_input_label, seg_input_stop):

    sam_predictor.set_image(frame)
    choosen_point = np.array(seg_input_point)
    # print("choosen_point", choosen_point)
    choosen_label = np.array(seg_input_label)  # 标签， 1是前景点用于选中，2是背景点用于排斥
    # print("choosen_label", choosen_label)
    sam_tmp_masks, sam_tmp_scores, sam_tmp_logits = sam_predictor.predict(
        point_coords=choosen_point,
        point_labels=choosen_label,
        multimask_output=True,
    )
    # sam_all_mask_tmp_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="checkpoints/sam_vit_h_4b8939.pth"))
    # sam_all_tmp_masks = sam_all_mask_tmp_generator.generate(frame)
    return sam_tmp_masks, sam_tmp_scores, sam_tmp_logits  # , sam_all_tmp_masks


# 定义鼠标点击事件
def mouse_click(event, mouse_x, mouse_y, flags, param):  # 鼠标点击事件
    # 全局变量，输入点，响应信号
    global seg_input_point_mouse, seg_input_label_mouse, seg_input_stop_mouse, center_x_mouse, center_y_mouse
    if not seg_input_stop_mouse:  # 判定标志是否停止输入响应了！
        if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键
            seg_input_point_mouse.append([mouse_x, mouse_y])
            seg_input_label_mouse.append(1)  # 1表示前景点
            seg_input_stop_mouse = True  # 定义语义分割的输入状态
            center_x_mouse = mouse_x
            center_y_mouse = mouse_y
        elif event == cv2.EVENT_RBUTTONDOWN:  # 鼠标右键
            seg_input_point_mouse.append([mouse_x, mouse_y])
            seg_input_label_mouse.append(0)  # 0表示后景点
            seg_input_stop_mouse = True
            center_x_mouse = mouse_x
            center_y_mouse = mouse_y
        elif event == cv2.EVENT_FLAG_LBUTTON:  # 鼠标左键长按 重置
            seg_input_point_mouse = []
            seg_input_label_mouse = []
            seg_input_stop_mouse = True
            center_x_mouse = mouse_x
            center_y_mouse = mouse_y


def apply_pointcloud_mask(depth_image, mask):
    return np.where(mask == 1, depth_image, 0)


def apply_color_mask(apply_color_tmp_image, apply_color_tmp_mask, apply_color_tmp_color):  # 对掩码进行赋予颜色
    color_dark = 0.6
    # print("apply_color_tmp_mask", apply_color_tmp_mask)
    for c in range(3):
        # np.where(condition, x, y)  满足condition取x,不满足取y
        apply_color_tmp_image[:, :, c] = np.where(
            apply_color_tmp_mask == 1,
            apply_color_tmp_image[:, :, c] * (1 - color_dark) + color_dark * apply_color_tmp_color[c],
            apply_color_tmp_image[:, :, c])
    return apply_color_tmp_image


def show_pose_o3d_pcl(lin_shi_ceshi_o3d_vs_pc_tmp, coordinate, attention_normal):
    """
    求解机器人的预调整姿态
    """
    attention_t = coordinate  # 提示点相对于相机坐标系原点的坐标 （圆心）
    new_axis_z = attention_normal  # 提示点处的法线方向，该法线方向为原坐标系的Z轴方向
    # print("new_axis_z的模", np.linalg.norm(new_axis_z))
    attention_mol = np.linalg.norm(attention_t)  # 求解距离范数
    attention_t_2 = attention_t - attention_mol * attention_normal  # 求解新的相机坐标原点 （圆弧末端点） （0，0，0 圆弧起始点）
    # print("相机的新的原点attention_t_2", attention_t_2)
    axis_pcd_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2,
                                                                   origin=attention_t_2)  # 坐标轴1
    axis_pcd_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2,
                                                                   origin=attention_t_2)  # 坐标轴2
    # print("attention_q_old:", attention_q_old)
    new_axis_x = np.zeros(3)
    new_axis_x = np.cross(attention_t / attention_mol, new_axis_z)
    new_axis_x = new_axis_x / np.linalg.norm(new_axis_x)
    new_axis_y = np.zeros(3)
    new_axis_y = -np.cross(new_axis_x, new_axis_z)
    new_axis_y = new_axis_y / np.linalg.norm(new_axis_y)
    # print("new_axis_y的模", np.linalg.norm(new_axis_y))

    rotation_matrix = np.array([[new_axis_x[0], new_axis_y[0], new_axis_z[0]],
                                [new_axis_x[1], new_axis_y[1], new_axis_z[1]],
                                [new_axis_x[2], new_axis_y[2], new_axis_z[2]]])
    # print("预调整姿态的旋转矩阵: ", rotation_matrix)
    axis_pcd_2.rotate(rotation_matrix, center=attention_t_2)

    # 创建 Open3D 的 LineSet 对象来表示点和方向向量
    attention_normal_line_set = o3d.geometry.LineSet()
    attention_normal_line_set.points = o3d.utility.Vector3dVector(
        np.array([camera_coordinate, camera_coordinate - 0.5 * attention_normal]))  # +号是冲外，-号是冲里
    attention_normal_line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    o3d.visualization.draw_geometries(
        [lin_shi_ceshi_o3d_vs_pc_tmp, attention_normal_line_set, axis_pcd, axis_pcd_1,
         axis_pcd_2])  # , point_show_normal=True)


def two_dimensions_grasp_wrench(two_dimensoions_grasp_mask):
    # print("two_dimensoions_grasp_mask", two_dimensoions_grasp_mask)
    two_dimensoions_grasp_contours, two_dimensoions_grasp_cnt = cv2.findContours(two_dimensoions_grasp_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    moment_2d_grasp_contours = cv2.moments(two_dimensoions_grasp_contours[0])  # 计算第一条轮廓的各阶矩,字典形式，根据自己的激光点效果寻找轮廓
    moment_2d_grasp_center_x = int(moment_2d_grasp_contours["m10"] / moment_2d_grasp_contours["m00"])
    moment_2d_grasp_center_y = int(moment_2d_grasp_contours["m01"] / moment_2d_grasp_contours["m00"])

    return two_dimensoions_grasp_contours, moment_2d_grasp_center_x, moment_2d_grasp_center_y


def pose_status_callback(status):
    global pose_adjustment_status
    print("status", status.data)
    pose_adjustment_status = status.data


def grasp_status_callback(status):
    global grasp_status_info
    print("status", status.data)
    grasp_status_info = status.data


def laser_waiter_status_callback(status):
    global laser_callback_status
    print("激光等待完发送回信儿了")
    laser_callback_status = status.data


if __name__ == '__main__':

    # 姿态预调整状态的订阅信息，抓取结束，返回ok状态执行下一步工作
    pose_adjust_status = rospy.Subscriber('/auto_grasp/pose_adjustment', String, pose_status_callback)
    # 抓取状态的订阅信息，抓取结束，返回ok状态执行下一步工作
    grasp_execute_status = rospy.Subscriber('/auto_grasp/grasp_status', String, grasp_status_callback)
    laser_back_wait_status = rospy.Subscriber('/auto_grasp/laser_waiter_back', String, laser_waiter_status_callback)
    model = YOLO('yolov8_laser_epochs_500.pt')  # 加载一个官方的检测模型
    track_history = defaultdict(lambda: [])

    # 定义一些激光的全局变量
    seg_input_point_laser = []  # 定义语义分割的输入点（激光的）
    seg_input_label_laser = []  # 定义语义分割的输入标签，包括前景点、背景点
    seg_input_stop_laser = False  # 定义语义分割的输入状态
    seg_input_point_laser_tmp = []  # 定义临时变量
    seg_input_label_laser_tmp = []
    seg_input_point_laser_tmp_opencvshow = []

    # 定义一些全局变量
    seg_input_point_mouse = []  # 定义语义分割的输入点（鼠标的）
    seg_input_label_mouse = []  # 定义语义分割的输入标签，包括前景点、背景点
    seg_input_stop_mouse = False  # 定义语义分割的输入状态
    camera_coordinate = np.zeros(3)
    center_x_mouse, center_y_mouse = 0, 0
    cv2.namedWindow("Scene", cv2.WINDOW_NORMAL)  # 初始化界面
    cv2.resizeWindow("Scene", 640, 480)  # 调整界面尺寸
    cv2.setMouseCallback("Scene", mouse_click)  # 调用鼠标点击
    # cap = cv2.VideoCapture(4)  注释了这段对话，因为不用cv导入图像，改为realsense的sdk导入图像
    seg_tmp_masks = []  # 定义一个空mask用于存储被选中target目标的segment mask
    S_optimal = []
    profiler = Profiler()  # 时间规划器、记录整体运行时间
    profiler.start()
    k = 0  # 定义循环次数
    k_instrin_in = [[605.471, 0.0, 327.157], [0.0, 604.936, 247.894], [0.0, 0.0, 1.0]]  # 相机内参

    eye_to_hand_matrix_t = np.array([0.584938, -0.00543464, 0.611657])
    eye_to_hand_matrix_q = np.quaternion(0.134079425734169, -0.475498688002052, -0.826136103898296, 0.270966498185416)  # qw, qx, qy, qz
    eye_on_hand_t_matrix = np.array([0.0373408, 0.100744, -0.15574])  # 2024年3月1日
    eye_on_hand_q_matrix = np.quaternion(0.07655784608114653, 0.014197669585180192, 0.009373324380869241, 0.9969199883500266)

    try:
        while True:

            laser_pointer_start = []
            laser_pointer_direction = []
            no_error_mean_matrix = None
            angle_set = []
            # _1, frame = cap.read()  # cv2读取图像
            # return 1相机内参，2深度参数，3BRG图，4深度图，5深度彩色映射图，6对齐的彩色帧，7对齐的深度帧
            point_111 = o3d.geometry.PointCloud()
            point_222 = o3d.geometry.PointCloud()

            pre_time = time.time()
            if eyetohand_camera_status:
                color_intrin_1, depth_intrin_1, img_color_1, img_depth_1, depth_mapped_image_1, aligned_color_frame_1, \
                    aligned_depth_frame_1, d_to_c_extrin_1, c_to_d_extrin_1 = get_aligned_images_1()
                # 输出标定板的图像和多个标定板下激光笔的姿态矩阵，姿态矩阵的平移部分基本相同，旋转部分按照y轴的60度依次旋转
                img_color_1_1 = copy.deepcopy(img_color_1)
                arucomarker_image, corners_set, matrix_set = Aruco_marker(img_color_1_1)  # 返回的是带aruco的图像和激光笔在相机中的姿态矩阵
                # point1 = get_pointcloud(img_color_1, img_depth_1, 1)  # 获取并保存局部点云图像
                """
                点云展示，通过这个看一看基于cv2.solvePnP识别的姿态精度
                """
                # axis_1_0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                # axis_2_0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2,
                #                                                              origin=matrix_set[0][:3, 3])
                # axis_2_0.rotate(matrix_set[0][:3, :3], center=matrix_set[0][:3, 3])
                # axis_3_0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2,
                #                                                              origin=matrix_set[1][:3, 3])
                # axis_3_0.rotate(matrix_set[1][:3, :3], center=matrix_set[1][:3, 3])
                # o3d.visualization.draw_geometries([point1, axis_1_0, axis_2_0, axis_3_0])

                # print("1.基于cv2.solvePnP计算的激光笔上标定板的矩阵(基于相机坐标系):\n", matrix_set)
                for corners in corners_set:
                    corners_center = np.mean(corners[0], axis=0)
                    corners_center_distance, corners_center_3d_coordinate = get_3d_camera_coordinate(corners_center,
                                                                                                     aligned_depth_frame_1,
                                                                                                     depth_intrin_1)
                    # print("角点的坐标:\n", corners_center)
                    # print("利用各角点的的像素和相机的深度信息计算的坐标:\n", corners_center_3d_coordinate)
                # draw_dynamic(matrix_set, i)    # 不想显示动态图注释即可
                eye_to_hand_rotation_matrix = quaternion.as_rotation_matrix(eye_to_hand_matrix_q)
                # comparative_rotation_matrix = quaternion_to_rotation_matrix(quaternion_from_rxyz)
                eye_to_hand_matrix = np.eye(4)
                eye_to_hand_matrix[:3, :3] = eye_to_hand_rotation_matrix
                eye_to_hand_matrix[:3, 3] = eye_to_hand_matrix_t
                # print("matrix_set:\n", matrix_set)
                if len(matrix_set) > 0:
                    matrix_set_after_eyetohand = np.matmul(eye_to_hand_matrix, matrix_set)
                    # print("matrix_set_after_eyetohand:\n", matrix_set_after_eyetohand)
                    # 然后就是往eye_on_hand相机上甩坐标变换
                    jaco_matrix = call_kinova_pose()
                    jaco_r_matrix = jaco_matrix[:3, :3]
                    jaco_t_matrix = jaco_matrix[:3, 3]
                    jaco_inv = np.eye(4)
                    jaco_inv[:3, :3] = jaco_r_matrix.T
                    jaco_inv[:3, 3] = -1 * np.matmul(jaco_r_matrix.T, jaco_t_matrix)
                    marker_after_jaco = np.matmul(jaco_inv, matrix_set_after_eyetohand)  # 左乘jaco矩阵
                    eye_on_hand_r_matrix = quaternion.as_rotation_matrix(eye_on_hand_q_matrix)
                    eye_on_hand_matrix = np.eye(4)
                    eye_on_hand_matrix[:3, :3] = eye_on_hand_r_matrix
                    eye_on_hand_matrix[:3, 3] = eye_on_hand_t_matrix
                    # marker_after_eyeonhand = np.matmul(eye_on_hand_matrix, transformation_marker_after)
                    eye_on_hand_matrix_inv = np.eye(4)
                    eye_on_hand_matrix_inv[:3, :3] = eye_on_hand_r_matrix.T
                    eye_on_hand_matrix_inv[:3, 3] = -1 * np.matmul(eye_on_hand_r_matrix.T, eye_on_hand_t_matrix)
                    marker_after_eyeonhand = np.matmul(eye_on_hand_matrix_inv, marker_after_jaco)  # 左乘
                    laser_pointer_in_eyeonhand = copy.deepcopy(marker_after_eyeonhand)
                    mean_matrix = np.mean(laser_pointer_in_eyeonhand, axis=0)
                    """基于marker标定板求解对齐的补偿矩阵"""
                    # error_matrix = np.array([[0.99356, -0.11225, 0.015322, 0.024902],
                    #                          [0.11058, 0.99027, 0.084456, -0.057204],
                    #                          [-0.024653, -0.082218, 0.99631, -0.006843],
                    #                          [0, 0, 0, 1]])

                    """基于colored_icp匹配求解的补偿矩阵"""
                    error_matrix = np.array([[0.99361, -0.11176, -0.015692, 0.029649],
                                             [0.11261, 0.99101, 0.072261, -0.049581],
                                             [0.0074747, -0.073566, 0.99726, 0.0041463],
                                             [0, 0, 0, 1]])

                    """无补偿矩阵"""
                    # error_matrix = np.eye(4)
                    no_error_mean_matrix = np.matmul(error_matrix, mean_matrix)
                    laser_pointer_direction = no_error_mean_matrix[:3, 1]  # 激光点的发射方向
                    laser_pointer_start = no_error_mean_matrix[:3, 3]  # 激光电的发射起始点
                    # print("2.眼在手外的矩阵:\n", eye_to_hand_matrix)
                    # print("3.激光笔标定板经过眼在手外后的矩阵(基于机械臂基座坐标系):\n", matrix_set_after_eyetohand)
                    # print("4.机械臂基座到机械手的姿态矩阵的逆矩阵:\n", jaco_inv)
                    # print("5.激光笔标定板经过眼在手外和基座姿态变换的矩阵(基于机械手末端坐标系):\n", marker_after_jaco)
                    # print("6.眼在手上矩阵的逆矩阵:\n", eye_on_hand_matrix_inv)
                    # print("7.激光笔标定板经过眼在手外、机械臂姿态变换逆、眼在手上逆后的平均矩阵:\n", mean_matrix)
                    # point2 = copy.deepcopy(point1)
                    # point3 = point2.rotate(eye_to_hand_matrix[:3, :3], center=(0,0,0))
                    # point3 = point2.translate(eye_to_hand_matrix[:3, 3])
                    # point4 = point3.rotate(jaco_inv[:3, :3], center=(0, 0, 0))
                    # point4 = point3.translate(jaco_inv[:3, 3])
                    # point5 = point4.rotate(eye_on_hand_matrix_inv[:3, :3], center=(0, 0, 0))
                    # point5 = point4.translate(eye_on_hand_matrix_inv[:3, 3])
                    # point5.transform(error_matrix)
                    # point_111 = copy.deepcopy(point5)

                    """
                    点云展示，通过这个看一看基于cv2.solvePnP识别的转换后的姿态精度
                    """
                    # axis_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                    # axis_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=laser_pointer_in_eyeonhand[0][:3, 3])
                    # axis_2.rotate(laser_pointer_in_eyeonhand[0][:3, :3], center=laser_pointer_in_eyeonhand[0][:3, 3])
                    # axis_3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=laser_pointer_in_eyeonhand[1][:3, 3])
                    # axis_3.rotate(laser_pointer_in_eyeonhand[1][:3, :3], center=laser_pointer_in_eyeonhand[1][:3, 3])
                    # o3d.visualization.draw_geometries([point5, axis_1, axis_2, axis_3])
            else:
                arucomarker_image = np.full((resolution_y, resolution_x, 3), 255, dtype=np.uint8)
                # draw the ArUco marker ID on the frame
                text_1 = "Waiting for eye-to-hand camera connection"
                cv2.putText(arucomarker_image, text_1, (20, int(resolution_y / 2)),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            1, (0, 0, 0), 2)

            if eyeonhand_camera_status:
                color_intrin_2, depth_intrin_2, img_color_2, img_depth_2, depth_mapped_image_2, aligned_color_frame_2, \
                    aligned_depth_frame_2, d_to_c_extrin_2, c_to_d_extrin_2 = get_aligned_images_2()
                yolov8track_image, laser_yolo_boxes = yolov8_detect(img_color_2)  # 获取yolo识别的激光电
                # yolov8track_image, laser_yolo_boxes = yolov8_track(img_color_2)    # 获取yolo识别的激光电
                # yolov8track_image, yolov8_corners_set, yolov8_matrix_set = Aruco_marker(img_color_2)   # 返回的是带aruco的图像和激光笔在相机中的姿态矩阵
                # print("8.基于cv2.solvePnP计算的激光笔上标定板的矩阵(基于eyetohand相机坐标系):\n", yolov8_matrix_set)
                # point6 = get_pointcloud(img_color_2, img_depth_2, 2)  # 获取并保存局部点云图像
                # point_222 = copy.deepcopy(point6)
                # laser_yolo_boxes = None
                if laser_yolo_boxes is not None:  # 如果识别到了超过一个激光点，那意味着出现了反光点
                    # print("yolov8_track函数返回后的激光边框", laser_yolo_boxes)
                    laser_3d_coordinate_set = []
                    laser_boxes = laser_yolo_boxes
                    # print("激光点的边框", laser_boxes)
                    for laser_box in laser_boxes:  # 遍历激光点
                        depth_pixel = [laser_box[0], laser_box[1]]
                        laser_3d_coordinate_distance, laser_3d_coordinate = get_3d_camera_coordinate(depth_pixel,
                                                                                                     aligned_depth_frame_2,
                                                                                                     depth_intrin_2)  # 求解激光电对应的三维坐标
                        laser_3d_coordinate_set.append(laser_3d_coordinate)  # 坐标填入集合
                    # print("laser_3d_coordinate_set", laser_3d_coordinate_set)
                    if len(laser_pointer_start) > 0 and len(laser_3d_coordinate_set) > 0:  # 这个是判断标定板的识别是否丢失的
                        laser_3d_coordinate_set_vector = np.array(laser_3d_coordinate_set) - np.array(
                            laser_pointer_start)  # 与标定的坐标相减计算向量
                        # print("eyeonhand上激光点三维坐标:\n", laser_3d_coordinate_set)
                        # print("从eyetohand转过来的激光笔发射起始坐标:\n", laser_pointer_start)
                        # print("eyeonhand上的激光点与eyetohand激光笔起始点构造的发射向量:\n", laser_3d_coordinate_set_vector)
                        # print("eyetohand上激光笔的发射方向:\n", laser_pointer_direction)
                        for laser_3d_coordinate_vector in laser_3d_coordinate_set_vector:  # 遍历向量
                            angle = angle_between_vectors(laser_3d_coordinate_vector, laser_pointer_direction)  # 计算夹角
                            angle_set.append(angle)
                        # print("angle_set", angle_set)
                        min_value_index = np.argmin(angle_set)
                        if angle_set[min_value_index] < 5:
                            output = open("angle_set.txt", "a")
                            # new_record_coordinate = " ".join(str(i) for i in angle_set)  # 将每个元素用空格分隔开
                            new_record_coordinate = angle_set[min_value_index]  # 保存预测的主视点的角度阈值
                            output.write(str(new_record_coordinate))
                            output.write("\n")
                            output.close()
                            x, y, w, h = laser_boxes[min_value_index]
                            pt1 = np.array([int(x - w), int(y - h)])
                            pt2 = np.array([int(x + w), int(y + h)])
                            cv2.rectangle(yolov8track_image, pt1, pt2, color=(0, 255, 0), thickness=4)
                            center = [int(x), int(y)]
                            seg_input_point_laser_tmp.append(center)  # 收集激光点，生成数组
                            seg_input_label_laser_tmp.append(1)  # 1表示前景点
                            seg_input_point_laser_tmp_opencvshow.append(center)
            else:
                yolov8track_image = np.full((resolution_y, resolution_x, 3), 255, dtype=np.uint8)
                text_2 = "Waiting for eye-on-hand camera connection"
                cv2.putText(yolov8track_image, text_2, (20, int(resolution_y / 2)),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            1, (0, 0, 0), 2)


            """
            stop信息初始化为False, 当收集超过100个激光点，进入if处理，对收集的100个激光点清空保留center,并将处理以后的信息状态发布出去，
            此时，stop状态为True,if即使收集超过100个激光点也进不来，并进入else进行清理。当激光信号等待回来，进入下一阶段姿态预处理，stop
            重新变为false,
            """
            # seg_input_stop_laser = True
            if not seg_input_stop_laser and len(np.array(seg_input_label_laser_tmp)) > 30:
                print("激光交互模块")
                # 添加GMM信号处理方法
                center_mean = gmm(seg_input_point_laser_tmp)
                print("seg_input_point_laser_tmp", seg_input_point_laser_tmp)
                # center_mean_1 = np.mean(np.array(seg_input_point_laser_tmp), axis=0)
                center_mean_1 = np.array(seg_input_point_laser_tmp)[29]
                print("GMM预测后的结果是：", center_mean)
                print("没有GMM预测后的结果是：", center_mean_1)
                center_x_mouse = int(center_mean[0])
                center_y_mouse = int(center_mean[1])
                seg_input_stop_laser = True    # 重置激光点处理状态
                seg_input_label_mouse.append(1)
                seg_input_stop_mouse = False    # 进入下一个基于点的姿态处理过程

                del seg_input_point_laser_tmp[:]    # 重置激光信息收集状态
                del seg_input_label_laser_tmp[:]
                if laser_callback_status == "laser_back_is_waiting":   # 等待发布时候,准备发布
                    print("激光信息收集完毕进行发布")
                    laser_topic_status = String()
                    laser_topic_status.data = "laser_is_ok"
                    laser_pub.publish(laser_topic_status)
                    rospy.loginfo("laser_topic_status is published: %s", laser_topic_status)
            elif laser_callback_status == "laser_back_is_working":   # 进入工作后，进来多少激光点都删掉
                # print("模块2")
                del seg_input_point_laser_tmp[:]    # 重置激光信息收集状态
                del seg_input_label_laser_tmp[:]

            # 2.鼠标交互, 鼠标中点左键就是前景点1，鼠标中点右键就是后景点标签0。
            if (seg_input_stop_mouse and len(np.array(seg_input_label_mouse)) > 0) or (laser_callback_status == "laser_back_is_ok"):
                print("模块3")
                if seg_input_stop_laser:
                    print("激光交互信号接入")
                    seg_input_stop_laser = False  # 重置激光点处理状态
                    laser_callback_status = "laser_back_is_working"
                    del seg_input_point_laser_tmp_opencvshow[:]  # 激光信号接入后就删除激光光斑的展示

                else:
                    print("屏幕交互信号接入")
                    laser_callback_status = "laser_back_is_working"

                # decision = input("是否为想要选择的对象，想要输入:y,想删除输入:n:")
                decision = "y"
                if decision == "y":
                    print("关注点为：", [center_x_mouse, center_y_mouse])
                    seg_input_stop_mouse = False  # 重置进入该判断的状态
                    lin_shi_ceshi_o3d_vs_pc = get_pointcloud(yolov8track_image, img_depth_2, 2)  # 获取并保存局部点云图像
                    # o3d.visualization.draw_geometries([lin_shi_ceshi_o3d_vs_pc])
                    # lin_shi_ceshi_o3d_vs_pc.paint_uniform_color([0,1,0])
                    pc = rs.pointcloud()
                    pc.map_to(aligned_color_frame_2)
                    points = pc.calculate(aligned_depth_frame_2)
                    vtx = np.asanyarray(points.get_vertices())
                    npy_vtx = np.zeros((len(vtx), 3), float)
                    for i in range(len(vtx)):
                        npy_vtx[i][0] = np.float64(vtx[i][0])
                        npy_vtx[i][1] = np.float64(vtx[i][1])
                        npy_vtx[i][2] = np.float64(vtx[i][2])

                    pcd_pc_vs_o3d = o3d.geometry.PointCloud()
                    pcd_pc_vs_o3d.points = o3d.utility.Vector3dVector(npy_vtx)
                    # points.paint_uniform_color([1,0,0])
                    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])  # 坐标轴
                    # o3d.visualization.draw_geometries([pcd_pc_vs_o3d, lin_shi_ceshi_o3d_vs_pc, axis_pcd])
                    # 设置法线长度
                    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                    lin_shi_ceshi_o3d_vs_pc.estimate_normals(search_param=search_param)
                    # print(seg_input_point)
                    # print("center_x_mouse, center_y_mouse", center_x_mouse, center_y_mouse)

                    dis, camera_coordinat = get_3d_camera_coordinate([center_x_mouse, center_y_mouse],
                                                                     aligned_depth_frame_2,
                                                                     depth_intrin_2)
                    camera_coordinate = np.array(camera_coordinat)
                    # print("鼠标x，y", [center_x_mouse, center_y_mouse])
                    normals_set = np.asarray(lin_shi_ceshi_o3d_vs_pc.normals)
                    points_set = np.asarray(lin_shi_ceshi_o3d_vs_pc.points)
                    # print("法线的个数", len(np.asarray(lin_shi_ceshi_o3d_vs_pc.normals)))
                    # print("点的坐标", np.asarray(lin_shi_ceshi_o3d_vs_pc.points))
                    # 使用 np.where() 来找到一维数组在二维数组中的位置
                    matches = np.where((points_set == camera_coordinate).all(axis=1))  # 找到关注点的对象的法线
                    matches_index = 0
                    if len(matches[0]) > 0:
                        print(f"一维数组在二维数组中的索引为：{matches[0][0]}")
                    else:
                        print("无法直接找到法线的方向，需要进一步处理")
                        new_points_set = points_set - camera_coordinate
                        # print("新点集是：", new_points_set)
                        new_points_set_norm = np.linalg.norm(new_points_set, axis=1)
                        # print("求范数后的新点集", new_points_set_norm)
                        matches_index = np.argmin(new_points_set_norm)
                        matches_min = np.min(new_points_set_norm)
                        # print(f"最小值是：{matches_min}")
                        # print(f"一维数组在二维数组中的索引为：{matches_index}")

                    attention_normals = np.array(normals_set[matches_index])  # 关注点处的法线是这个
                    print("关注点法线的方向三维向量", attention_normals)
                    print("关注点的三维坐标_相对于相机坐标系", camera_coordinate)
                    prompt_info.normal = attention_normals
                    prompt_info.coordinate = camera_coordinate
                    print(prompt_info)
                    prompt_pub.publish(prompt_info)
                    rospy.loginfo("Prompt is published: %s", prompt_info)
                    show_pose_o3d_pcl(lin_shi_ceshi_o3d_vs_pc, camera_coordinate, attention_normals)    # 展示调整后的坐标系姿态

                    """
                    计算相机坐标系变换后，激光点相对应的新像素点用于图像分割
                    """
                    new_camera_frame_coord = camera_coordinate
                    new_camera_frame_normal = attention_normals
                    new_axis_z = new_camera_frame_normal  # 提示点处的法线方向，该法线方向为原坐标系的Z轴方向
                    attention_mol = np.linalg.norm(new_camera_frame_coord)  # 求解距离范数
                    attention_t_2 = new_camera_frame_coord - attention_mol * new_camera_frame_normal  # 求解新的相机坐标原点 （圆弧末端点） （0，0，0 圆弧起始点）
                    # attention_q_old = vector_to_quaternion(new_camera_frame_normal)  # 求解法线的四元数
                    # print("attention_q_old:", attention_q_old)
                    new_axis_x = np.zeros(3)
                    new_axis_x = np.cross(new_camera_frame_coord / attention_mol, new_axis_z)
                    new_axis_x = new_axis_x / np.linalg.norm(new_axis_x)
                    new_axis_y = np.zeros(3)
                    new_axis_y = -np.cross(new_axis_x, new_axis_z)
                    new_axis_y = new_axis_y / np.linalg.norm(new_axis_y)

                    attention_r_matrix = np.array([[new_axis_x[0], new_axis_y[0], new_axis_z[0]],
                                                   [new_axis_x[1], new_axis_y[1], new_axis_z[1]],
                                                   [new_axis_x[2], new_axis_y[2], new_axis_z[2]]])
                    # print("预调整姿态的旋转矩阵: ", attention_r_matrix)
                    attention_matrix = np.eye(4)  # 这个矩阵存在很大的问题20240301！回答：经测试不存在问题20240319
                    # print("矩阵的转置", attention_r_matrix.T)
                    # print("矩阵的逆矩", np.linalg.inv(attention_r_matrix))
                    attention_matrix[:3, :3] = attention_r_matrix  # 2024/03/01 现在这个代码好像是对的
                    attention_matrix[:3, 3] = np.array(attention_t_2)  # 得加个括号，双括号才行, 好像不加也行

                    attention_matrix_inv = np.eye(4)
                    attention_matrix_inv[:3, :3] = attention_r_matrix.T
                    attention_matrix_inv[:3, 3] = -1 * np.matmul(attention_r_matrix.T, np.array(attention_t_2))
                    camera_coordinate_with_1 = np.ones(4)
                    camera_coordinate_with_1[0:3] = camera_coordinate
                    print("camera_coordinate_with_1", camera_coordinate_with_1)
                    camera_coordinate_base_on_new_frame = np.matmul(attention_matrix_inv, camera_coordinate_with_1)
                    print("原始相机坐标系的像素点", [center_x_mouse, center_y_mouse])
                    new_attention_pixel = rs.rs2_project_point_to_pixel(color_intrin_2, camera_coordinate_base_on_new_frame[:3])
                    print("变换后相机坐标系像素点", new_attention_pixel)
                    color_by_intrinsic = np.matmul(k_instrin_in, np.array(camera_coordinate).T)
                    # print("color_by_intrinsic", color_by_intrinsic)
                    color_point = rs.rs2_transform_point_to_point(d_to_c_extrin_2, camera_coordinate)  #
                    color_pixel = rs.rs2_project_point_to_pixel(color_intrin_2, color_point)
                    # print("三维转回来后的点：", color_pixel)

                else:
                    del seg_input_point_mouse[:]
                    del seg_input_label_mouse[:]
                    center_x_mouse = 0.0
                    center_y_mouse = 0.0
                    seg_input_stop_mouse = False  # 定义语义分割的输入状态
                    laser_callback_status = "laser_back_is_waiting"

            if pose_adjustment_status == "ok":    # ！！！！！需要加一些信息，看看center在这个过程中的保持程度
                # print("等1s，让延迟的图像进来")
                time.sleep(1)
                print("进入SAM分割阶段")
                input_point = [[int(new_attention_pixel[0]), int(new_attention_pixel[1])]]
                print("变换后的输入点", input_point)
                input_label = [1]
                input_bool = True  # 定义语义分割的输入状态
                thread_sam = MyThread(SAM_wrapper, (yolov8track_image,
                                                    input_point,
                                                    input_label,
                                                    input_bool))
                # print("seg_input_label_mouse", seg_input_label_mouse)
                thread_sam.start()
                thread_sam.join()
                sam_masks, sam_scores, sam_logits = thread_sam.get_result()
                seg_tmp_masks = sam_masks[0]    # 0,1,2分割的逐渐变大
                local_depth_2 = apply_pointcloud_mask(img_depth_2, seg_tmp_masks)
                local_pcl_points = get_pointcloud(yolov8track_image, local_depth_2, 2)  # 获取并保存局部点云图像
                # o3d.visualization.draw_geometries([local_pcl_points, attention_normal_line_set, axis_pcd])
                o3d.visualization.draw_geometries([local_pcl_points, axis_pcd])

                del seg_input_point_mouse[:]
                del seg_input_label_mouse[:]

                seg_input_stop_mouse = False  # 定义语义分割的输入状态
                print("分割完毕")

            if len(seg_tmp_masks) != 0:
                # print("目标图像掩码上色")
                # color = tuple(np.array([255, 255, 255]).tolist())
                color = tuple(np.random.randint(0, 256, 3).tolist())  # 设置颜色随机
                yolov8track_image = apply_color_mask(yolov8track_image, seg_tmp_masks, color)  # 为实例分割物品赋予阴影
                # img_white_color = apply_white_mask(img_color, seg_tmp_masks, color)   # 为实例分割物品赋予阴影
                seg_tmp_masks = np.where(seg_tmp_masks == True, 1, 0)  # 改成黑白图， 有阴影部分为白1，无阴影部分为黑0
                seg_tmp_masks = np.array(seg_tmp_masks, np.uint8)  # 改为int8
                grasp_2d_wrench_contours, grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y = two_dimensions_grasp_wrench(
                    seg_tmp_masks)  # 求解阴影的边界和圆心
                grasp_2d_wrench_contours_reshape = np.reshape(grasp_2d_wrench_contours[0], (-1, 2))
                grasp_2d_wrench_contours_center = np.array(
                    [grasp_2d_wrench_contours_center_x, grasp_2d_wrench_contours_center_y])
                # np.savetxt("grasp_stability/txt/boundary.txt", grasp_2d_wrench_contours_reshape, fmt='%s')
                # np.savetxt("grasp_stability/txt/center.txt", grasp_2d_wrench_contours_center, fmt='%s')
                # print(grasp_2d_wrench_contours_reshape)
                # print(grasp_2d_wrench_contours_center)

                cv2.drawContours(yolov8track_image, grasp_2d_wrench_contours, 0, color, 2)  # 绘制轮廓，填充（图像，轮廓，轮廓序号，颜色，轮廓线粗细）

                # 将这两个if全部注释可以隐藏抓取
                if pose_adjustment_status == "ok" and len(S_optimal) == 0:
                    print("力闭合分析")
                    q_max, W_max, S_optimal, S_rotate_degree = grasp_wrench_2d(grasp_2d_wrench_contours_reshape,
                                                                               grasp_2d_wrench_contours_center)
                    S_optimal = S_optimal.astype(np.int32)

                if len(S_optimal) != 0:
                    color_2 = tuple(np.random.randint(0, 256, 3).tolist())  # 设置颜色随机
                    cv2.line(yolov8track_image, S_optimal[0], S_optimal[1], color_2, 1, cv2.LINE_AA)
                    cv2.circle(yolov8track_image, S_optimal[0], 8, color_2, -1)
                    cv2.circle(yolov8track_image, S_optimal[1], 8, color_2, -1)

                if pose_adjustment_status == "ok" and len(S_optimal) != 0:
                    print("计算抓取点")
                    print("左侧接触像素点：", S_optimal[1])    # 旋转回来后，S_optimal[1]变成了左侧接触点，转过来了
                    print("右侧接触像素点：", S_optimal[0])
                    # cv2.arrowedLine(img_color, S_optimal[0] + S_rotate_degree, S_optimal[0], color=(0, 255, 0), thickness=2, tipLength=0.3)
                    dis_left, touch_left = get_3d_camera_coordinate([S_optimal[0][0] - 15, S_optimal[0][1]],
                                                                    # 先加后减能让左点更靠里
                                                                    aligned_depth_frame_2,
                                                                    depth_intrin_2)
                    # print("dis_left", dis_left, "m")
                    print("左侧接触实际三维点", touch_left, "m")
                    dis_right, touch_right = get_3d_camera_coordinate([S_optimal[1][0] + 15, S_optimal[1][1]],
                                                                      aligned_depth_frame_2,
                                                                      depth_intrin_2)
                    # print("dis_right", dis_right, "m")
                    print("右侧接触实际三维点", touch_right, "m")
                    camera_xyz = np.array(touch_left + touch_right) / 2  # 收集相机坐标系下中心坐标点的xyz坐标

                    angle_vector = S_optimal[1] - S_optimal[0]  # 求解旋转角度的向量
                    angle = 0
                    # 求解需要旋转的角度
                    if angle_vector[0] == 0 and angle_vector[1] > 0:  # 角度为正无穷
                        angle = np.pi / 2
                    elif angle_vector[0] == 0 and angle_vector[1] < 0:  # 角度为负无穷
                        angle = -np.pi / 2
                    elif angle_vector[0] == 0 and angle_vector[1] == 0:  # 这意味着两个点重合了
                        angle = 0
                    elif angle_vector[0] != 0:
                        oriented_vector = np.array([angle_vector[1] / angle_vector[0]])
                        angle = np.arctan(oriented_vector)  # 如果求解结果为负，机械手末端应该逆时针旋转。如果求解结果为正，机械手末端应该顺时针旋转。
                        print("机械臂在抓取之前应该旋转的角度为：", angle)

                    touch_left.append(1)
                    touch_right.append(1)
                    if (touch_left[2] or touch_right[2]) > 1.5:
                        print("touch_left[2]", touch_left[2])
                        print("touch_right[2]", touch_right[2])
                        break

                    print("订阅之前的grasp_info", grasp_info)
                    grasp_info.left_touch = touch_left
                    grasp_info.right_touch = touch_right
                    grasp_info.angle_touch = angle
                    print("发布抓取信息")
                    print("发布之后的grasp_info", grasp_info)
                    time.sleep(1)
                    grasp_pub.publish(grasp_info)
                    # rospy.loginfo("grasp is published: %s %s %s",
                    #               grasp_info.left_touch, grasp_info.right_touch, grasp_info.angle_touch)
                    pose_adjustment_status = ""
                    cv2.imwrite("history/grasp_{}.png".format(time_time), yolov8track_image)

            if grasp_status_info == "ok":
                print("grasp status is ok")
                seg_tmp_masks = []
                laser_callback_status = "laser_back_is_waiting"
                center_x_mouse = 0.0
                center_y_mouse = 0.0
                grasp_status_info = ""
                S_optimal = []

            # if move_status == "ok":
            #     print("ok", ok)

            cv2.circle(yolov8track_image, [center_x_mouse, center_y_mouse], 2, (0, 0, 255), -1)
            # print("图像运行")
            # if len(seg_input_point_laser_tmp_opencvshow) < 31:
            for i in range(len(seg_input_point_laser_tmp_opencvshow)):
                cv2.circle(yolov8track_image, seg_input_point_laser_tmp_opencvshow[i], 1, (0, 255, 0), -1)

            time_time = time.time()
            # cv2.imshow("Scene", np.hstack((img_color, img_color_222)))  # 展示彩色图像和深度图像
            cv2.imshow("Scene", yolov8track_image)  # 展示图像
            # cv2.imwrite("laser/edge_{}.png".format(time_time), img_color)
            # mix = cv2.addWeighted(img_color, 0.8, depth_mapped_image, 0.2, 0)
            # cv2.imshow("Scene", mix)  # 展示图像
            # ori_frame = img_color  # 储存上一帧
            k = k + 1
            key = cv2.waitKey(1)

            # del seg_input_point[:]
            # del seg_input_label[:]

            if key == 27 or (key & 0XFF == ord("q")):
                cv2.destroyAllWindows()
                break
    finally:
        # destroy the instance
        # cap.release()
        pipeline_1.stop()
        pipeline_2.stop()


    profiler.stop()
    profiler.print()
