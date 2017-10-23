
# データ加工・処理・分析モジュール
import numpy as np
import pandas as pd

def set_time(dataframe, col_name):
    '''
    to_datetimeを使うための前処理
    '''
    dataframe[col_name] = dataframe[col_name].map(lambda x : transform_time(x))
    return dataframe


# In[3]:


def transform_time(x):
    '''
    set_time内で使う関数
    to_datetimeで24時をサポートしないので00に変更する処理
    '''
    str_x = str(x)
    res = ''
    if str(x)[8:10] == '24':
        res = str_x[0:4] + '-' + str_x[4:6] + '-' + str_x[6:8] + ' 00:'+str_x[10:12] 
    else:
        res = str_x[0:4] + '-' + str_x[4:6] + '-' + str_x[6:8] + ' '+ str_x[8:10] +':'+str_x[10:12]
    return res


# In[4]:


def normalize_array(x):
    '''
    min, max, min-max正規化を行なった配列(np.array)を返す
    '''
    x = np.array(x)
    x_min = x.min()
    x_max = x.max()
    normalized = (x - x_min) / (x_max - x_min) 
    return x_min, x_max, normalized


# In[5]:


def denormalize_array(normalized_x, x_min, x_max):
    '''
    正規化前のmin, maxを用いて元のスケールに戻す
    '''
    normalized_x = np.array(normalized_x)
    denormalize_array = (normalized_x) * (x_max - x_min) + x_min
    return denormalize_array


# In[8]:d

def get_chunked_data(x, chunk_size):
    '''
    rnnに突っ込むためのchunk_sizeごとに区切った系列データを作る
    '''
    input_list = []
    for i in range(0, len(x) - chunk_size + 1):
        input_list.append(x[i : i + chunk_size])
    input_list = np.array(input_list)
    return input_list


# In[9]:

def drop_nan(X, Y):
    '''
    正解データがnanであるデータの組を削除
    '''
    mask = np.isnan(Y)
    X = X[~mask]
    Y = Y[~mask]
    return X, Y

def get_dataset(amd_data, output_data, chunk_size):
    '''
    学習用と予測用にデータを分割
    '''
    # 日射量の欠損値を一つ前の値で置換/output_data
    amd_data['sl'] = amd_data['sl'].fillna(method='bfill')
    amd_data['max_tp'] = amd_data['max_tp'].fillna(method='bfill')

    # 学習に必要なデータ
    # 2012/01/01 00:10 ~ 2015/12/30 20:00のamdデータを用いて
    # 2012/01/03 03:50 ~ 2015/12/31 23:50のデータを予測する
    train_x_startID = amd_data[amd_data['datetime'] == pd.to_datetime('2012-01-01 00:10')].index[0]
    train_x_endID = amd_data[amd_data['datetime'] == pd.to_datetime('2015-12-30 20:00')].index[0]

    train_y_endID = amd_data[amd_data['datetime'] == pd.to_datetime('2015-12-31 23:50')].index[0]
    lapse = train_y_endID - train_x_endID
    train_y_startID = train_x_startID + lapse + chunk_size -1

    train_amd_data = amd_data[['sl', 'max_tp']][train_x_startID:(train_x_endID+1)]
    train_output_data = np.array(output_data[train_y_startID:(train_y_endID+1)])

    # 予測に必要なデータ
    # 2015/12/29 20:30 ~ 2017/3/30 20:00のamdデータを用いて
    # 2016/01/01 00:00 ~ 2017/3/31 23:50のoutputデータを予測する
    test_y_startID = amd_data[amd_data['datetime'] == pd.to_datetime('2016-01-01 00:00')].index[0]
    test_startID = test_y_startID - lapse - chunk_size + 1
    test_endID = amd_data[amd_data['datetime'] == pd.to_datetime('2017-3-30 20:00')].index[0]

    test_amd_data = amd_data[['sl', 'max_tp']][test_startID:(test_endID+1)]

    return train_amd_data, train_output_data, test_amd_data

def prepare_rnn(train_amd_data, test_amd_data, train_output_data, chunk_size):
    '''
    rnnに突っ込むための準備を行う
    '''
    # rnnに突っ込むためにmin-max正規化しておく
    normalized_amd = (train_amd_data - train_amd_data.min()) / (train_amd_data.max() - train_amd_data.min())
    normalized_amd = np.array(normalized_amd)
    normalized_test_amd = (test_amd_data - test_amd_data.min()) / (test_amd_data.max() - test_amd_data.min())
    normalized_test_amd = np.array(normalized_test_amd)

    #時系列データのリストにする
    input_list = get_chunked_data(normalized_amd, chunk_size)

    # outputがnanである学習ペアを取り除く
    input_list, train_output_data = drop_nan(input_list, train_output_data)

    # outputのmin_max正規化
    output_min, output_max, normalized_output = normalize_array(train_output_data)
    print("output_min : " + str(output_min) + ", output_max : " + str(output_max))

    # testデータの入力を用意
    test_input_list = get_chunked_data(normalized_test_amd, chunk_size)

    # RNNに突っ込むためにデータを整形
    X = np.array(input_list).reshape(len(input_list), chunk_size, input_list.shape[2])
    Y = np.array(normalized_output).reshape(len(input_list), 1)
    X_predict = np.array(test_input_list).reshape(len(test_input_list), chunk_size, test_input_list.shape[2])

    return X, Y, X_predict, output_min, output_max