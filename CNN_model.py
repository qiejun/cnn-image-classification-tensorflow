import tensorflow as tf
from data import *

"""
定义Model类并进行训练
"""

class Model(object):

    def __init__(self,output_num,lr,epochs,train_data_file,test_data_file):
        self.epochs = epochs
        self.lr = lr
        self.input_x = tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)
        self.input_y = tf.placeholder(shape=[None,],dtype=tf.int64)
        self.output_num = output_num
        self.logits, self.output_with_softmax = self.net()
        self.loss, self.acc, self.optimizer, self.merge= self.train_op()
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file

    def net(self):
        x = tf.layers.conv2d(self.input_x,8,3,(1,1),padding='same',activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x,2,2,'same')
        x = tf.layers.conv2d(x,16,3,(1,1),'same',activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x,2,2,'same')
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x,512,activation=tf.nn.relu)
        x = tf.layers.dropout(x,0.5)
        logits = tf.layers.dense(x,self.output_num)
        output_with_softmax = tf.nn.softmax(logits)
        return logits,output_with_softmax

    def train_op(self):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,1),self.input_y),tf.float32))
        tf.summary.scalar('loss', loss) #绘制loss曲线
        tf.summary.scalar('accuracy',acc) #绘制accuracy曲线
        merge = tf.summary.merge_all()
        return loss,acc,optimizer,merge

    def train(self,sess):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_wirter = tf.summary.FileWriter(logdir='log/train',graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir='log/test',graph=sess.graph)
        iters = 0
        avg_loss = 0
        for epoch in range(self.epochs):
            test_batch_num = 0
            for train_x, train_y in train_data(self.train_data_file,32,(128,128)):
                _ = sess.run(self.optimizer, feed_dict = {self.input_x:train_x,self.input_y:train_y})
                train_loss, train_accuracy, train_summary = sess.run([self.loss,self.acc,self.merge],feed_dict={self.input_x:train_x,self.input_y:train_y})
                train_wirter.add_summary(train_summary,iters)
                avg_loss = avg_loss+train_loss
                if iters%5 == 0 and iters!=0:
                    print('epoch:',epoch,',iters:',iters,',train loss:',avg_loss/5)
                    avg_loss = 0
                if iters%50==0 :
                    test_x, test_y, num_batch= test_data(self.test_data_file,64,test_batch_num,(128,128))
                    test_loss, test_accuracy, test_summary = sess.run([self.loss,self.acc,self.merge],feed_dict={self.input_x:test_x,self.input_y:test_y})
                    test_writer.add_summary(test_summary,iters)
                    print('epoch:',epoch,',iters:',iters,',test loss:',test_loss,',test accuracy:',test_accuracy)
                    test_batch_num = test_batch_num+1
                    if test_batch_num==num_batch:
                        test_batch_num=0
                iters = iters+1
                if iters%200==0 and iters!=0:
                    saver.save(sess=sess,save_path='model/model.ckpt'+str(iters))

def main():
    model = Model(output_num=2, lr=0.0001, epochs=20,train_data_file='E:\\test_data\\train',test_data_file='E:\\test_data\\test')
    sess = tf.Session()
    model.train(sess)
    # print(model.output_with_softmax.name)
    # print(model.input_x.name)
    sess.close()

if __name__=='__main__':
    main()