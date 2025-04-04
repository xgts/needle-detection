import os

import numpy as np
import torch
import cv2
from ptflops import get_model_complexity_info

from needle_detection_train.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from needle_detection_train.cfg.default import get_cfg_defaults
from Test_kal.util import pred_lines_1


def Line_point(image_path,needle_map):
    model_path=r''
    cfg = get_cfg_defaults()
    model = MobileV2_MLSD_Large(cfg).cuda().eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    img_fn = image_path
    img1 = cv2.imread(img_fn)
    img = cv2.resize(img1, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lines = pred_lines_1(img, model, [512, 512], 0.1, 20,needle_map)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if (lines==[]):
        return [0,0]
    else:
        for l in lines:
            return [int(l[2]),int(l[3])]

    for l in lines:
        print([int(l[0]/512*720),int(l[1]/512*440)])
        cv2.circle(img1, [int(l[0]/512*720),int(l[1]/512*440)], 2, (0, 0, 255))
        cv2.imshow('line_img', img1)
        # cv2.imshow("image",output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    path = r''
    path_save = r''

    namelist = sorted(os.listdir(path))
    for name1 in namelist:
        print(name1)
        path_image = rf"{path}/{name1}"
        path_label = path_image.replace('image', 'mask')

        #point = Line_point(path_image, name1)
        #print(point)

