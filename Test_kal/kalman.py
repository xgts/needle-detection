import numpy as np
import os
import cv2
import torch

from mask_point import *
from Pre import Line_point
def updata_trace_list(box_center, trace_list, max_list_len=50):
    if len(trace_list) <= max_list_len:
        trace_list.append(box_center)
    else:
        trace_list.pop(0)
        trace_list.append(box_center)
    return trace_list

def kalman(path):
    mask_list = sorted(os.listdir(rf'{path}/mask'))

    mask_path=os.path.join(path,'mask',mask_list[0])
    initial_target_box=point(mask_path)
    # if(initial_target_box==[0,0]):
    #     offset_x=0
    #     offset_y=0
    # else:
    #     offset_x = np.random.randint(-20, 21)
    #     offset_y = np.random.randint(-20, 21)

    # tip_x = int(np.clip(initial_target_box[0] + offset_x, 0, 511))
    # tip_y = int(np.clip(initial_target_box[1] + offset_y, 0, 511))
    #initial_state = np.array([[tip_x,tip_y, 0, 0]]).T  # [中心x,中心y,宽w,高h,dx,dy]
    initial_state=np.array([[initial_target_box[0],initial_target_box[1],0,0]]).T

    # 状态转移矩阵，上一时刻的状态转移到当前时刻
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # 状态观测矩阵
    H = np.eye(4)

    # 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
    # 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
    # Q = np.eye(6) * 0.1
    Q = np.eye(4) * 5
    # 观测噪声协方差矩阵R，p(v)~N(0,R)
    # 观测噪声来自于检测框丢失、重叠等
    R = np.eye(4) * 0.1

    # 控制输入矩阵B
    B = None
    # 状态估计协方差矩阵P初始化
    P = np.eye(4)
    name_list = sorted(os.listdir(f'{path}/image'))
    X_posterior = np.array(initial_state)  # 后验估计
    P_posterior = np.array(P)
    Z = np.array(initial_state)
    trace_list = []

    for img in name_list:
        img_path = os.path.join(f'{path}/image', img).replace('\\', '/')
        last_box_posterior = X_posterior[0:2]

        # image = cv2.imread(img_path)
        # cv2.circle(image, [int(last_box_posterior[0] / 512 * 720), int(last_box_posterior[1] / 512 * 440)], 2, (0, 255, 0))
        # path_save=os.path.join(path,'Pre_position')
        # if not os.path.exists(path_save):
        #     os.makedirs(path_save)
        # cv2.imwrite(os.path.join(path_save, img), image)
        # cv2.imshow("pre",image)
        # cv2.waitKey(0)
        # ---------使用分割结果观测值------------

        needle_tip_map = np.zeros((1,1, 512, 512), dtype=np.float32)
        needle_tip_map[0,0, int(np.clip(last_box_posterior[1],0,511)), int(np.clip(last_box_posterior[0],0,511))] = 255
        needle_tip_map[0,0, :, :] = cv2.GaussianBlur(needle_tip_map[0,0, :, :], (3, 3), 0.0)
        needle_tip_map[0,0, :, :] = np.array(needle_tip_map[ 0,0, :, :], dtype=np.float32) / 255.0



        xy = Line_point(img_path,needle_tip_map)
        # map = np.zeros((440,720), dtype=np.float32)
        # map[ int(np.clip(last_box_posterior[1]/512*440, 0, 439)), int(np.clip(last_box_posterior[0]/512*720, 0, 729))] = 255
        # map[:, :] = cv2.GaussianBlur(map[:, :], (3, 3), 0.0)
        # map[map != 0] = 255
        # map = map.astype(np.uint8)
        #
        # path_save = os.path.join(path, 'Pre_map')
        # if not os.path.exists(path_save):
        #     os.makedirs(path_save)
        # cv2.imwrite(os.path.join(path_save, img), map)
        # print(xywh)
        # exit()
        target_box = xy
        if (target_box[0] == 0 & target_box[1] == 0):  # 1000
            max_iou_matched = False
        else:
            max_iou_matched = True

        if max_iou_matched == True:
            # 如果有分割结果,则认为该框为观测值
            box_center = (target_box[0], target_box[1])
            trace_list = updata_trace_list(box_center, trace_list, 1000)

            # 计算dx,dy
            dx = xy[0] - X_posterior[0]
            dy = xy[1] - X_posterior[1]

            Z[0:2] = np.array([xy]).T
            Z[2::] = np.array([dx, dy])

        if max_iou_matched:
            # -----进行先验估计-----------------
            X_prior = np.dot(A, X_posterior)
            box_prior = X_prior[0:2]


            # -----计算状态估计协方差矩阵P--------
            P_prior_1 = np.dot(A, P_posterior)
            P_prior = np.dot(P_prior_1, A.T) + Q

            # ------计算卡尔曼增益---------------------
            k1 = np.dot(P_prior, H.T)
            k2 = np.dot(np.dot(H, P_prior), H.T) + R
            K = np.dot(k1, np.linalg.inv(k2))

            # --------------后验估计------------
            X_posterior_1 = Z - np.dot(H, X_prior)
            X_posterior = X_prior + np.dot(K, X_posterior_1)

            box_posterior = X_posterior[0:2]
            # ---------更新状态估计协方差矩阵P-----
            P_posterior_1 = np.eye(4) - np.dot(K, H)
            P_posterior = np.dot(P_posterior_1, P_prior)
        else:
            # 如果预测失败，此时失去观测值，那么直接使用上一次的最优估计作为先验估计
            # 此时直接迭代，不使用卡尔曼滤波
            # X_posterior = np.dot(A, X_posterior)
            # box_posterior = X_posterior[0:2]
            # box_center = ((int(box_posterior[0])), (int(box_posterior[1])))
            # trace_list = updata_trace_list(box_center, trace_list, 1000)
            # # print(box_center[0])
            # if box_center[0] > 550:
            #     continue
            trace_list = updata_trace_list([0,0], trace_list, 1000)

    return trace_list


if __name__ == '__main__':

    path=r''
    trace_list=kalman(path)
    print(trace_list)

    img_list=sorted(os.listdir(os.path.join(path,"image")))


    i=0
    for name in img_list:
        img_path=os.path.join(path,"image",name)

        img=cv2.imread(img_path)
        cv2.circle(img, [int(trace_list[i][0]/512*720), int(trace_list[i][1]/512*440)], 2, (0, 0, 255))
        i=i+1
        print(name)
        path_save=os.path.join(path,'baseline+kal')
        # if not os.path.exists(path_save):
        #     os.makedirs(path_save)
        # cv2.imwrite(os.path.join(path_save, name), img)
        cv2.imshow('image', img)
        cv2.waitKey(0)