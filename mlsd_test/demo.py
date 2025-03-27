import torch
import os
import sys
import cv2

from models.mbv2_mlsd_large import MobileV2_MLSD_Large
from  cfg.default import  get_cfg_defaults
from util import  pred_lines,pred_lines_1

#import matplotlib.pyplot as plt
def pre(image_path,name):
    current_dir = os.path.dirname(__file__)
    if current_dir == "":
        current_dir = "./"
    # model_path = current_dir+'/models/mlsd_tiny_512_fp32.pth'
    # model = MobileV2_MLSD_Tiny().cuda().eval()

    # model_path = current_dir + '/models/mlsd_large_512_fp32.pth'
    # model = MobileV2_MLSD_Large().cuda().eval()


    model_path=r''
    cfg = get_cfg_defaults()
    model = MobileV2_MLSD_Large(cfg).cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    img_fn=image_path
    img = cv2.imread(img_fn)
    img = cv2.resize(img, (512, 512))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lines = pred_lines_1(img, model, [512, 512], 0.1, 20)   #_1不带分割
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_1 = cv2.imread(img_fn,1)
    img_2 = cv2.imread(img_fn, 1)

    for l in lines:
        cv2.line(img_1, (int(l[0]/512*720), int(l[1]/512*440)), (int(l[2]/512*720), int(l[3]/512*440)), (0,0,255), 1,16)
        cv2.circle(img_2, [int(l[2] / 512 * 720), int(l[3] / 512 * 440)], 2, (0, 0, 255))


    #cv2.circle(img_2,[int(l[0]/512*720), int(l[1]/512*440)],2,(0,0,255))
    cv2.imshow("image", img_2)
    cv2.imshow('out',img_1)
    cv2.waitKey(0)
    # plt.imshow(img_1)
    # plt.show()
    # path2=rf""
    # if not os.path.exists(path2):

    #     os.makedirs(path2)
    # cv2.imwrite(os.path.join(path2,name),img)

if __name__ == '__main__':
    path=r''
    namelist=sorted(os.listdir(path))
    for name1 in namelist:
        path1=rf"{path}/{name1}"
        pre(path1,name1)

