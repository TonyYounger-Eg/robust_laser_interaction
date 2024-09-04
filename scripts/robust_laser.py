# coding:utf-8
#
# stackoverflow.py
#
#  Created on: 2024/5/22
#      Author: Tex Yan Liu
#
# description: 对其进行一系列改错后(https://stackoverflow.com/questions/72702953/attributeerror-module-cv2-aruco-has-no-attribute-drawframeaxes)
#              仍然无法生成标定板相应的姿态
#              他这个链接里面的内容，错误挺多的
#              本代码中的内容已经开始逐步实现各项功能
#              本代码中将筛选掉非88号标定板的噪音数据
#              添加上kinova_jaco的调用，但是由于订阅无法并行处理，需要调试,等机械臂回来再试
#              继续写了当识别多个marker矩阵并转换后取矩阵结果平均值的方法
#              基本功能都可以实现了，但是由于用的是Yolo追踪算法，导致存在不正确的现象，在版本6里面进行更改
#              版本6中的各项矩阵通过调用点云进行验证了，矩阵都是正确的，变换也是正确的，就是精度太低。引入open3d点云库后，法线cv2.Rodrigues在某种情况下是正确的,并且误差也没那么大，6是整个框架的较为成熟版本
#              7 是准备对成熟的6开始修改测试了，先测试两个相机看同一个标定版减少误差的方法，误差估计补偿矩阵已经检测成功，现在把补偿矩阵放置于正式使用中
#              8 确实证明了加入补偿矩阵能够帮助提高激光对齐的准度阈值

import copy
import time

import cv2  # 惊天大BUG，cv2必须在torch上面，否则会RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
import pyrealsense2 as rs  # 导入realsense的sdk模块
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import quaternion
from geometry_msgs.msg import PoseStamped
import rospy
import threading
import open3d as o3d


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
    print(rgbd_device_1)
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
    print(rgbd_device_2)
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
    x = get_coord_tmp_depth_pixel[0]
    y = get_coord_tmp_depth_pixel[1]
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


if __name__ == '__main__':
    rospy.init_node('yolo_and_marker', anonymous=True)
    cv2.namedWindow("img_color_1", cv2.WINDOW_NORMAL)  # 初始化界面
    cv2.resizeWindow("img_color_1", resolution_x, resolution_y)  # 调整界面尺寸

    # Store the track history
    track_history = defaultdict(lambda: [])

    # 加载官方或自定义模型
    model = YOLO('yolov8_laser_epochs_500.pt')  # 加载一个官方的检测模型
    # model = YOLO('yolov8n-seg.pt')  # 加载一个官方的分割模型
    # model = YOLO('yolov8n-pose.pt')  # 加载一个官方的姿态模型
    # x, y, z, rx, ry, rz
    eye_to_hand_matrix_t = np.array([0.584938, -0.00543464, 0.611657])
    # eye_to_hand_matrix_r = np.array([-2.52826, 0.0361606, 2.10852])
    # quaternion_from_rxyz = Rxyz_to_Quaternion(eye_to_hand_matrix_r)
    # print(quaternion_from_rxyz)
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
                arucomarker_image, corners_set, matrix_set = Aruco_marker(img_color_1_1)   # 返回的是带aruco的图像和激光笔在相机中的姿态矩阵
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
                    corners_center_distance, corners_center_3d_coordinate = get_3d_camera_coordinate(corners_center, aligned_depth_frame_1, depth_intrin_1)
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
                    marker_after_jaco = np.matmul(jaco_inv, matrix_set_after_eyetohand)   # 左乘jaco矩阵
                    eye_on_hand_r_matrix = quaternion.as_rotation_matrix(eye_on_hand_q_matrix)
                    eye_on_hand_matrix = np.eye(4)
                    eye_on_hand_matrix[:3, :3] = eye_on_hand_r_matrix
                    eye_on_hand_matrix[:3, 3] = eye_on_hand_t_matrix
                    # marker_after_eyeonhand = np.matmul(eye_on_hand_matrix, transformation_marker_after)
                    eye_on_hand_matrix_inv = np.eye(4)
                    eye_on_hand_matrix_inv[:3, :3] = eye_on_hand_r_matrix.T
                    eye_on_hand_matrix_inv[:3, 3] = -1 * np.matmul(eye_on_hand_r_matrix.T, eye_on_hand_t_matrix)
                    marker_after_eyeonhand = np.matmul(eye_on_hand_matrix_inv, marker_after_jaco)   # 左乘
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
                    laser_pointer_direction = no_error_mean_matrix[:3, 1]    # 激光点的发射方向
                    laser_pointer_start = no_error_mean_matrix[:3, 3]     # 激光电的发射起始点
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
                cv2.putText(arucomarker_image, text_1, (20, int(resolution_y/2)),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            1, (0, 0, 0), 2)

            if eyeonhand_camera_status:
                color_intrin_2, depth_intrin_2, img_color_2, img_depth_2, depth_mapped_image_2, aligned_color_frame_2, \
                    aligned_depth_frame_2, d_to_c_extrin_2, c_to_d_extrin_2 = get_aligned_images_2()
                yolov8track_image, laser_yolo_boxes = yolov8_detect(img_color_2)    # 获取yolo识别的激光电
                # yolov8track_image, laser_yolo_boxes = yolov8_track(img_color_2)    # 获取yolo识别的激光电
                # yolov8track_image, yolov8_corners_set, yolov8_matrix_set = Aruco_marker(img_color_2)   # 返回的是带aruco的图像和激光笔在相机中的姿态矩阵
                # print("8.基于cv2.solvePnP计算的激光笔上标定板的矩阵(基于eyetohand相机坐标系):\n", yolov8_matrix_set)
                # point6 = get_pointcloud(img_color_2, img_depth_2, 2)  # 获取并保存局部点云图像
                # point_222 = copy.deepcopy(point6)
                # laser_yolo_boxes = None
                if laser_yolo_boxes is not None:   # 如果识别到了超过一个激光点，那意味着出现了反光点
                    # print("yolov8_track函数返回后的激光边框", laser_yolo_boxes)
                    laser_3d_coordinate_set = []
                    laser_boxes = laser_yolo_boxes
                    # print("激光点的边框", laser_boxes)
                    for laser_box in laser_boxes:     # 遍历激光点
                        depth_pixel = [laser_box[0], laser_box[1]]
                        laser_3d_coordinate_distance, laser_3d_coordinate = get_3d_camera_coordinate(depth_pixel, aligned_depth_frame_2, depth_intrin_2)   # 求解激光电对应的三维坐标
                        laser_3d_coordinate_set.append(laser_3d_coordinate)    # 坐标填入集合
                    # print("laser_3d_coordinate_set", laser_3d_coordinate_set)
                    if len(laser_pointer_start) > 0 and len(laser_3d_coordinate_set) > 0:    # 这个是判断标定板的识别是否丢失的
                        laser_3d_coordinate_set_vector = np.array(laser_3d_coordinate_set) - np.array(laser_pointer_start)    # 与标定的坐标相减计算向量
                        # print("eyeonhand上激光点三维坐标:\n", laser_3d_coordinate_set)
                        # print("从eyetohand转过来的激光笔发射起始坐标:\n", laser_pointer_start)
                        # print("eyeonhand上的激光点与eyetohand激光笔起始点构造的发射向量:\n", laser_3d_coordinate_set_vector)
                        # print("eyetohand上激光笔的发射方向:\n", laser_pointer_direction)
                        for laser_3d_coordinate_vector in laser_3d_coordinate_set_vector:     # 遍历向量
                            angle = angle_between_vectors(laser_3d_coordinate_vector, laser_pointer_direction)    # 计算夹角
                            angle_set.append(angle)
                        # print("angle_set", angle_set)
                        min_value_index = np.argmin(angle_set)
                        if angle_set[min_value_index] < 5:
                            output = open("angle_set.txt", "a")
                            # new_record_coordinate = " ".join(str(i) for i in angle_set)  # 将每个元素用空格分隔开
                            new_record_coordinate = angle_set[min_value_index]   # 保存预测的主视点的角度阈值
                            output.write(str(new_record_coordinate))
                            output.write("\n")
                            output.close()
                            x, y, w, h = laser_boxes[min_value_index]
                            pt1 = np.array([int(x - w), int(y - h)])
                            pt2 = np.array([int(x + w), int(y + h)])
                            cv2.rectangle(yolov8track_image, pt1, pt2, color=(0, 255, 0), thickness=4)
            else:
                yolov8track_image = np.full((resolution_y, resolution_x, 3), 255, dtype=np.uint8)
                text_2 = "Waiting for eye-on-hand camera connection"
                cv2.putText(yolov8track_image, text_2, (20, int(resolution_y/2)),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            1, (0, 0, 0), 2)

            """
            二合一总的点云展示,通过这个判断最后eyetohand点云经过变换后到eyeonhand的姿态的准确度
            """
            # if no_error_mean_matrix is not None:
            #     axis_all_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
            #     axis_all_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=no_error_mean_matrix[:3, 3])
            #     axis_all_2.rotate(no_error_mean_matrix[:3, :3], center=no_error_mean_matrix[:3, 3])
            #     o3d.visualization.draw_geometries([point_111, point_222, axis_all_1, axis_all_2])
            # cv2.imshow("img_color_1", np.hstack((img_color_1, img_color_2)))  # 展示彩色图像和深度图像
            current_time = time.time()
            fps = 1 / (current_time - pre_time)
            print("fps", fps)
            cv2.imshow("img_color_1", np.hstack((arucomarker_image, yolov8track_image)))  # 展示彩色图像和深度图像
            # cv2.imshow("img_color_1", yolov8track_image)  # 展示彩色图像和深度图像
            key = cv2.waitKey(1)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # destroy the instance
        # cap.release()
        pipeline_1.stop()



