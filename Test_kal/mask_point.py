import cv2
import numpy as np
def point(label_path):
    idx_y=0
    img=cv2.imread(label_path,1)
    array=np.sum(img[:,:,0],axis=0)
    point=[1000,1000]
    for i in range(len(array)):
        if(array[i]>0):
            idx_y=i
            point[0]=i
            break
    for i in range(img.shape[0]):
        if(img[i,idx_y,0]>0):
            point[1]=i
            break

    # cv2.circle(img, point, 2, (0, 0, 255))
    # cv2.imshow('gt_img', img)
    # # cv2.imshow("image",output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if (point[0] == 1000 & point[1] == 1000):
        return [0,0]
    else:
        point=[point[0]/720*512,point[1]/440*512]
        return point
