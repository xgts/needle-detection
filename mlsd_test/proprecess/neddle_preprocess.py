import os
import cv2
import numpy as np
import json
def point(label_path):
    img=cv2.imread(label_path,1)
    array=np.sum(img[:,:,0],axis=0)
    for i in range(len(array)):
        if(array[i]>0):
            idx_y=i
            break
    for i in range(img.shape[0]):
        if(img[i,idx_y,0]>0):
            idx_x=i
            break
    for j in range(len(array)-1,0,-1):
        if(array[j]>0):
            idx_y1=j
            break
    for k in range(img.shape[0]-1,0,-1):
        if(img[k,idx_y1,0]>0):
            idx_x1=k
            break
    point=[idx_y,idx_x,idx_y1,idx_x1]
    if point[2]>580:
        point[2]=580
    return point

if __name__ == '__main__':
    result=[]
    path = r''
    image_path=os.path.join(path,'image')
    label_path=os.path.join(path,'label')

    label_list=os.listdir(label_path)
    for label_name in label_list:
        label_path1=os.path.join(label_path,label_name)
        image_path1=os.path.join(image_path,label_name).replace('.bmp','.png')

        filename=label_name.replace('.bmp','.png')
        lines=point(label_path1)
        height=cv2.imread(image_path1).shape[0]
        width=cv2.imread(image_path1).shape[1]

        item={
            "filename":filename,
            "lines":[lines],
            "height":height,
            "width":width
        }
        result.append(item)

    json_path=os.path.join(path,'valid.json')
    with open(json_path,'w') as json_file:
        json.dump(result,json_file)


