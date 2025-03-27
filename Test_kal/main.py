import os

import cv2
import numpy as np
import json

from ptflops import get_model_complexity_info
from torchvision.transforms import transforms
from kalman import *
from mask_point import *
from meter import AverageMeter





def line_len_fn(out, gt):
    if(out[0]==0 and out[1]==0 and gt[0]==0 and gt[1]==0):
        return 0
    else:
        len1 = np.sqrt((out[0] - gt[0]) ** 2 + (out[1] - gt[1]) ** 2)
        return len1

totensor = transforms.ToTensor()

def floader_result_kal(path):
    img_list=sorted(os.listdir(rf'{path}/image'))
    Line_avg=AverageMeter()
    kal_result=kalman(path)
    i=0
    error_floader=[]
    for img_name in img_list:
        img_path=os.path.join(rf'{path}/image',img_name)
        mask_path = os.path.join(rf'{path}/mask', img_name)
        if not os.path.exists(mask_path):
            mask_path=mask_path.replace('png','bmp')

        image = cv2.imread(img_path)

        #cv2.circle(image, [int(kal_result[i][0]/512*720),int(kal_result[i][1]/512*440)], 2, (0, 0, 255))#红

        target=point(mask_path)

        #cv2.circle(image, [int(target[0]/512*720),int(target[1]/512*440)], 2, (255, 0, 0))
        Line_len=line_len_fn(kal_result[i],target)
        error_floader.append(Line_len)
        i=i+1
        #cv2.imshow("image", image)
        #cv2.waitKey(0)
        Line_avg.update(Line_len,1)

    return Line_avg.avg,error_floader


def floader_success(path,weight):
    img_list = sorted(os.listdir(rf'{path}/image'))
    Line_success=0
    kal_result = kalman(path)
    i=0
    for img_name in img_list:
        img_path = os.path.join(rf'{path}/image', img_name)
        mask_path = os.path.join(rf'{path}/mask', img_name)
        if not os.path.exists(mask_path):
            mask_path=mask_path.replace('png','bmp')

        target = point(mask_path)
        Line_len = line_len_fn(kal_result[i], target)
        if(Line_len<weight):
            Line_success+=1
        i=i+1
    return Line_success/len(img_list)


if __name__ == '__main__':
    Test_path = r''
    #Test_H_good_path = rf'{Test_path}/TEST_H_good_2'
    Test_H_good_path = rf'{Test_path}/test_total'
    Test_H_sample_good_path = rf''

    Test_show_kal=rf''

    path = Test_H_good_path

    list_test = sorted(os.listdir(path))
    list_test = list(filter(lambda x: x.find("dle") > 0, list_test))
    # print(list_test)
    # exit()
    #list_test = ['needle_6', 'needle_7_1', 'needle_1_1', 'needle_12', 'needle_11_1']
    #list_test = ['needle_1', 'needle_5', 'needle_7', 'needle_13', 'needle_22', 'needle_23', 'needle_27', 'needle_29',
     #        'needle_30']

    line_total_avg_kal = AverageMeter()
    line_kal_mean = []

    error_total={}

    # 求success
    line_total_avg_success = AverageMeter()
    line_success = []
    for name in list_test:
        image_path = os.path.join(path, name)  # needle_1子文件夹地址

        line_floader_result_mean ,error_f= floader_result_kal(image_path)
        line_kal_mean.append(line_floader_result_mean)

        error_total[name]=error_f
        line_floader_result_success = floader_success(image_path, 15)
        line_success.append(line_floader_result_success)

        print(error_f)
        print("Name:", name, "Line_kal:", line_floader_result_mean)
        print("Name:", name, "Line_success:", line_floader_result_success)

        line_total_avg_kal.update(line_floader_result_mean)
        line_total_avg_success.update(line_floader_result_success)

    print(error_total)
    print(line_kal_mean)#每一个文件夹的平均值
    print(line_success)#每一个文件夹的成功率
    print("aug:", line_total_avg_kal.avg)
    print("success:", line_total_avg_success.avg)

