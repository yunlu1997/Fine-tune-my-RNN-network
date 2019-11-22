import numpy as np
import os
def subject_and_label(file_name):
    subject = file_name[-8:-6].strip('_')
    label = file_name[0]
    return subject, label

def data_len(data, WINDOW_LENGTH =600,stride =1):
    ROWS = -1
    for i in range(0, data.size, stride):
        ROWS += 1
        if data.size - i < WINDOW_LENGTH:
            return ROWS

def normalize(data):
    mx = max(data)
    mn = min(data)
    return [(float(i) - mn) / (mx - mn) for i in data]

def extract_data(data, label=1, WINDOW_LENGTH=600, stride=1):
    ROWS = data_len(data, WINDOW_LENGTH = WINDOW_LENGTH, stride= stride)
    row = 0
    COLS = 1 + WINDOW_LENGTH
    data_processed = np.zeros(shape=(ROWS, COLS))
    for i in range(0, data.size, stride):
        if data.size - i < WINDOW_LENGTH:
            break
        data_processed[row][0] = label
        data_processed[row][1:COLS] = normalize(data[i:WINDOW_LENGTH + i])
        row+=1
    return data_processed


if __name__ =='__main__':
    '''工作路径及后缀名'''
    workspace = 'F:/Breathing_GRU_Model/Sum_demo/raw_dataset'
    ext = 'csv'
    type_list = os.listdir(workspace)

    '''存储路径及名称'''
    store_dir = 'F:/Breathing_GRU_Model/Sum_demo/processed_dataset'
    store_name = 'test'
    csvname = os.path.join(store_dir, store_name) + '.' + ext

    '''时间窗口及步长设置'''
    WINDOW_LENGTH = 600 # 600, 1200, 1800
    stride = 200

    dataset_train = np.zeros(shape=(1, 1 + WINDOW_LENGTH))  # 数据集初始化，最后需要删除
    dataset_test  = np.zeros(shape=(1, 1 + WINDOW_LENGTH))  # 数据集初始化，最后需要删除

    for type in type_list:
        file_list = os.listdir(os.path.join(workspace, type))
        # print(file_list）
        for file in file_list:
            file_path = os.path.join(workspace, type, file)
            subject, label = subject_and_label(file)
            # print(subject)
            data = np.loadtxt(file_path, delimiter=',')
            # print(data)
            if data.size < WINDOW_LENGTH:  # 如果数据少于窗口长度，则忽略
                continue
            else:
                data_temp = extract_data(data, WINDOW_LENGTH=WINDOW_LENGTH, stride =stride, label=label)
                #data_temp_without_label = np.delete(data_temp, 0, 1)
                # 训练集取自 subject 1-3
                if int(subject) in [1, 2, 3]:
                    dataset_train = np.vstack((dataset_train, data_temp))
                    print('正在处理:' + file_path)
                    # 存储每个Subject的数据，不带label
                    np.savetxt(os.path.join(store_dir,str(WINDOW_LENGTH),'stride'+str(stride),file), data_temp, delimiter=',')
                # 测试集取自剩余
                else:
                    dataset_test = np.vstack((dataset_test, data_temp))
                    print('正在处理:' + file_path)
                    # 存储每个Subject的数据，不带label
                    np.savetxt(os.path.join(store_dir, str(WINDOW_LENGTH), 'stride'+str(stride), file), data_temp, delimiter=',')

     # 存储总的数据集，分为train 和 test。分别取自不同的subject
    dataset_train = np.delete(dataset_train, 0, 0)  # 删除用于初始化的第一行
    dataset_test = np.delete(dataset_test, 0, 0)  # 删除用于初始化的第一行
    np.savetxt(str(WINDOW_LENGTH)+'_'+str(stride)+'train'+'.csv', dataset_train, delimiter=',')
    np.savetxt(str(WINDOW_LENGTH) + '_' + str(stride) + 'test' + '.csv', dataset_test, delimiter=',')
