import numpy as np
import os
import pickle
import cv2

"""
制作训练数据
read_data():获取图片的地址以及标签
train_data():获取训练数据
test_data():获取测试数据
"""

def read_data(file):
    data_list = []
    label_list = []
    classes_name = os.listdir(file)
    class_dict = {label:index for index,label in enumerate(classes_name)} #生成字典,名字：类别
    cache_dict = open('pickle/class_dict.pickle','wb')
    pickle.dump(class_dict,cache_dict) #保存字典到pickle,方便以后查看
    cache_dict.close()
    for class_name in classes_name:
        imgs_path = os.path.join(file,class_name)
        imgs_name = os.listdir(imgs_path)
        for img_name in imgs_name:
            single_img_path = os.path.join(imgs_path,img_name)
            data_list.append(single_img_path)
            label_list.append(class_dict[class_name])
    Temp = np.array([data_list,label_list]).T
    np.random.shuffle(Temp)
    labels = list(Temp[:,1])
    datas = list(Temp[:,0])
    return datas,labels

def train_data(train_file,batch_size,resize_shape):
    datas, labels = read_data(train_file)
    num_batch = len(datas)//batch_size
    for i in range(num_batch):
        imgs = []
        train_datas = datas[batch_size*i:batch_size*(i+1)]
        train_lables = labels[batch_size*i:batch_size*(i+1)]
        for img_path in train_datas:
            img = cv2.imread(img_path)
            img = cv2.resize(img,resize_shape)
            img = img/255 #归一化处理
            imgs.append(img)
        yield np.array(imgs),np.array(train_lables)

def test_data(test_file,batch_size,batch_num,reshap_size):
    datas,labels = read_data(test_file)
    num_batch = len(datas)//batch_size
    test_datas = datas[batch_size*batch_num:batch_size*(batch_num+1)]
    test_labels = labels[batch_size*batch_num:batch_size*(batch_num+1)]
    imgs = []
    for img_path in test_datas:
        img = cv2.imread(img_path)
        img = cv2.resize(img,reshap_size)
        img = img/255
        imgs.append(img)
    return np.array(imgs),np.array(test_labels),num_batch