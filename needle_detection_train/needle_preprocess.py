import os
import cv2
import numpy as np
import json
def point(label_path):
    idx_y=0
    idx_y1=0
    idx_x=0
    idx_x1=0
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
    return point

if __name__ == '__main__':
    result=[]
    path = r'/Data/train'
    image_path=os.path.join(path,'image')
    label_path=os.path.join(path,'mask')

    label_list=os.listdir(label_path)
    for label_name in label_list:
        label_path1=os.path.join(label_path,label_name)
        image_path1=os.path.join(image_path,label_name)

        filename=label_name
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


    # # # 计算分割的索引
    # split_index = int(len(result——ablation) * 0.8)
    #
    # # 分割列表
    # train_data = result——ablation[:split_index]
    # val_data = result——ablation[split_index:]

    # 定义文件名
    train_filename = '/Data/train.json'
    #val_filename = /Data/val.json'

    # 写入训练数据到 JSON 文件
    with open(train_filename, 'w') as train_file:
        json.dump(result, train_file)

    # # 写入测试数据到 JSON 文件
    # with open(val_filename, 'w') as test_file:
    #     json.dump(val_data, test_file)

    print(f"训练数据已写入 {train_filename}")
    #print(f"测试数据已写入 {val_filename}")




