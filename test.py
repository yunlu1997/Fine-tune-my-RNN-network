import os
import generate_dataset
import numpy as np
import tensorflow as tf
from BI_AT_GRU import Model

success_num=0
sum_num=0

'''工作路径及后缀名'''
workspace = 'F:/Breathing_GRU_Model/Sum_demo'
data_dir ='processed_dataset'
window_size = 600
stride = 200
ext = 'csv'
file_list = os.listdir(os.path.join(workspace, data_dir, str(window_size),'stride'+str(stride)))
print(file_list)

'''存储路径'''
store_dir = 'F:/Breathing_GRU_Model/Sum_demo/results'

'''调用模型'''
sess = tf.Session()

def load_model():
    """
        Loading the pre-trained model and parameters.
    """
    global X, yhat
    modelpath = 'F:/Breathing_GRU_Model/Sum_demo/model/60/'
    saver = tf.train.import_meta_graph(modelpath + 'model.ckpt-1701.meta')
    saver.restore(sess, modelpath+'model.ckpt-1701')
    print('Successfully load the pre-trained model!')

def create_labels(data,label = 1):
    label_matrix = np.ones(shape=( data.shape[0]))
    label_matrix_ = label_matrix * label
    return label_matrix_

# 载入预先训练好的模型
load_model()
for file in file_list:
    file_path = os.path.join(workspace,data_dir,str(window_size),'stride'+str(stride),file)
    #print(file_path)
    label = int(file[0])
    number = file[-8:-4].strip('_')
    #print(label,number)
    data = np.loadtxt(file_path, delimiter=',')
    # 非二维数据则跳过
    if data.ndim != 2 :
        continue
    # 删除第一列的标签信息
    data = np.delete(data, 0,1)
    data = data[:, :, np.newaxis]
    print (data.shape)
    # 得到计算图以及input
    graph = tf.get_default_graph()
    #for op in graph.get_operations():
       # print(op.name)
    input = graph.get_tensor_by_name("input:0")
    labels = graph.get_tensor_by_name("labels:0")
    input_keep_prob = graph.get_tensor_by_name("input_keep_prob:0")
    output_keep_prob = graph.get_tensor_by_name("output_keep_prob:0")
    input_label = create_labels(data, label=label)
    feed_dict = {input: data, labels: input_label, input_keep_prob :1.0,output_keep_prob :1.0 }
    # 得到输出结果
    argmax = graph.get_tensor_by_name('output/ArgMax:0')
    prediction_matrix = sess.run(argmax, feed_dict=feed_dict)
    prediction_matrix_without_noise = prediction_matrix[prediction_matrix < 6]   # 只保留0-5的预测值
    #print(prediction_matrix)
    # 找出一个样本中出现的次数最多的预测类别,作为此样本的最终预测结果。
    if any(prediction_matrix_without_noise):
        final_prediction = np.argmax(np.bincount(prediction_matrix_without_noise))+ 1 # 输出的label是0-6，我们需要的是1-7，在这里加1。
        print(final_prediction)
        if final_prediction == label:
            success_num +=1
            print('预测成功！')
    sum_num += 1

    store_path = os.path.join(store_dir, str(label)+'_'+str(number)+'.csv')


    #print(store_path)
acc = success_num / sum_num
print('正确率:',acc)
sess.close()



