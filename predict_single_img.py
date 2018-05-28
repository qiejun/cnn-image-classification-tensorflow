import numpy as np
import tensorflow as tf
from PIL import Image
import pickle

"""
用训练好的模型对单张图像进行预测
输入：图像的地址
输出：预测结果，以及概率
"""

def read_img(img_file):
    img = Image.open(img_file)
    img = img.resize((128,128))
    img = np.array(img)/255
    img = np.expand_dims(img,axis=0)#扩展维度为四维
    return img

img = read_img('D:\\test_image\\people0826.png')
print(img.shape)

#加载标签字典
file = open('pickle/class_dict.pickle','rb')
class_dict = pickle.load(file)
file.close()

#为了方便图片预测显示，转换字典，key与value互换
class_dict = {x:y for y,x in class_dict.items()}

#加载训练好的模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('model/model.ckpt800.meta')
saver.restore(sess=sess,save_path='model/model.ckpt800')
output_with_softmax = tf.get_default_graph().get_tensor_by_name('Softmax:0')
input_x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')

#预测
output = sess.run(output_with_softmax,feed_dict={input_x:read_img('D:\\test_image\\people0826.png')})#输入预测图像的地址
output_index = np.int(np.argmax(output,axis=-1))
print('The predict is :',class_dict[output_index],' with score:',output[0][output_index])

sess.close()