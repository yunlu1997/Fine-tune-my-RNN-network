import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

"""输入的数据,输入对应的标签"""
data_all = np.loadtxt('test.csv', delimiter=',', dtype=float)  #总数据
print('样本行数', data_all.shape[0])

"""参数设置"""
seq_len = 600
n_inputs = 1

"""设定模型的路径"""
modelpath = 'F:/Breathing_GRU_Model/Sum_demo/model/60/'

"""恢复graph"""
saver = tf.train.import_meta_graph(modelpath + 'model.ckpt-1701.meta')
graph = tf.get_default_graph() # 如果只有一个会话时，会话中的图也就是默认的图，无需在Session中进行
ops1 = graph.get_operations()
#for i in ops1:
    #print(i.name) # 打印所有的操作

"""
    # 在Session开启前可以进行网络的改动，比如：
    output = graph.get_tensor_by_name('output/add:0')
    # 随意添加的网络的新分支
    c = output + tf.Variable([1.])
"""

"""恢复输入"""
input_tensor = graph.get_tensor_by_name("input:0")
labels = graph.get_tensor_by_name("labels:0")
input_keep_prob = graph.get_tensor_by_name("input_keep_prob:0")
output_keep_prob = graph.get_tensor_by_name("output_keep_prob:0")

"""恢复输出, 即单个样本的loss"""
# cross_entropy 即单个样本的loss
cross_entropy = graph.get_tensor_by_name('output/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0')
output_w = graph.get_tensor_by_name('output/output_w:0')  # for test
atn_w = graph.get_tensor_by_name('attention/atn_w:0')   # for test
argmax = graph.get_tensor_by_name('output/ArgMax:0') # for test
"""定义需要更新的output层的权重, freeze 其他层"""
tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'output')

"""训练过程，采用梯度下降，由于是单个样本训练，不需要剪辑梯度了"""
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, var_list=tvars)

with tf.Session() as sess:
    """恢复权重"""
    saver.restore(sess, modelpath + 'model.ckpt-1701')
    """开始训练，单独输入每一行，并人工输入标签"""
    for row in range(data_all.shape[0]-380):
        data = data_all[row]
        plt.plot(data)
        plt.show()
        data = data[np.newaxis, :, np.newaxis]
        label_matrix = np.ones(shape=(data.shape[0]))
        label_num = int(input("此段波形对应的标签是： （范围从0-6)"))
        label_matrix = label_matrix * label_num
        atn_w_, output_w_,predict ,_ = sess.run([atn_w, output_w, argmax,train_op],feed_dict={input_tensor: data, labels : label_matrix, input_keep_prob:1.0, output_keep_prob:1.0})
        print('测试，这个输出是输出层的权重变化，应该有值：\n', sess.run(output_w_ - output_w))
        print('测试，这个输出是注意力层的权重变化，应该全为0：\n', sess.run(atn_w_ - atn_w))
        print('真实标签：',label_matrix, '预测标签：',predict)
        print(row,'/', data_all.shape[0])


    """保存模型，关闭会话"""
    saver.save(sess, 'fine_tune_model/60/model.ckpt')
    sess.close()
# def load_graph(modelpath):
#     """
#         Loading the pre-trained model and parameters.
#         Return graph.
#     """
#     saver = tf.train.import_meta_graph(modelpath + 'model.ckpt-1701.meta')
#     saver.restore(sess, modelpath+'model.ckpt-1701')
#     graph = sess.graph
#     print('Successfully load the pre-trained model!')
#     return graph




# """加载预训练模型"""
# graph = load_model(modelpath)

"""
   查看恢复的模型参数
   tf.trainable_variables()查看的是所有可训练的变量；
   tf.global_variables()获得的与tf.trainable_variables()类似，只是多了一些非trainable的变量，比如定义时指定为trainable=False的变量；
   sess.graph.get_operations()则可以获得几乎所有的operations相关的tensor
"""
# tvs = [v for v in tf.trainable_variables()]
# print('获得所有可训练变量:')
# for v in tvs:
#     print(v.name)
#     # print(sess.run(v))

# gv = [v for v in tf.global_variables()]
# print('获得所有变量:')
# for v in gv:
#     print(v.name, '\n')
#     sess.graph.get_operations()

# """得到需要用到的变量"""
#
#
#
#     sess.close()

