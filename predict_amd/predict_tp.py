
# coding: utf-8

# # **前日の20:00までのデータを用いて翌日の00:00~23:50までの気温が予測する**

# ### **必要な関数・ライブラリ**

# In[1]:


# データ加工・処理・分析モジュール
import numpy as np
import pandas as pd
import random
import argparse
import os
import simpleLSTM

def set_time(dataframe, col_name):
    '''
    to_datetimeを使うための前処理
    '''
    dataframe[col_name] = dataframe[col_name].map(lambda x : transform_time(x))
    return dataframe

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

def normalize_array(x):
    '''
    min, max, min-max正規化を行なった配列(np.array)を返す
    '''
    x = np.array(x)
    x_min = x.min()
    x_max = x.max()
    normalized = (x - x_min) / (x_max - x_min) 
    return x_min, x_max, normalized

def denormalize_array(normalized_x, x_min, x_max):
    '''
    正規化前のmin, maxを用いて元のスケールに戻す
    '''
    normalized_x = np.array(normalized_x)
    denormalize_array = (normalized_x) * (x_max - x_min) + x_min
    return denormalize_array

def get_chunked_data(x, chunk_size):
    '''
    rnnに突っ込むためのchunk_sizeごとに区切った系列データを作る
    '''
    input_list = []
    for i in range(0, len(x) - chunk_size + 1):
        input_list.append(x[i : i + chunk_size])
    input_list = np.array(input_list)
    return input_list

def drop_nan(X, Y):
    '''
    正解データがnanであるデータの組を削除
    '''
    mask = np.isnan(Y)
    X = X[~mask]
    Y = Y[~mask]
    return X, Y

def calc_mae(X, Y):
    mse = 0
    for i in range(len(X)):
        mse += np.abs(X[i]- Y[i])
    return mse/len(X)

def main():
	# パーサーを作る
	parser = argparse.ArgumentParser(
			prog='predict output ', # プログラム名
            usage='predict output data by lstm', # プログラムの利用方法
            description='you can change place to predict', # 引数のヘルプの前に表示
            epilog='end', # 引数のヘルプの後で表示
            add_help=True, # -h/–help オプションの追加
            )

	parser.add_argument('-p', '--target_place', help='select place', type=int, required = True)
	parser.add_argument('-c', '--chunk_size', help='this parameter must be same learned model', type=int, required = True)
	parser.add_argument('-e', '--epochs', help='select epochs', type=int, required = False)

	# 入力引数
	args = parser.parse_args()

	target_place = args.target_place
	chunk_size = args.chunk_size
	epochs = 100
	if args.epochs:
		epochs = args.epochs

	model_name = "model_"+str(target_place)+"_chunk_"+str(chunk_size)+"_epoch_"+str(epochs)

	print("---- predict " + model_name + " ----")
	
	print("target_place : " + str(target_place) +  ", chunk_size :" + str(chunk_size))

	# ### **データの準備**
	print("---- load amd data ----")
	# アメダスデータの読み込み
	if target_place == 1 or target_place == 2:
	    # 横浜アメダスのデータを使って予測する, amd_46106
	    # 各amdidはamd_masterに記載されている
	    amd_data = pd.read_csv('data/raw_data/amd_46106.tsv', delimiter = '\t')
	    amd_data = set_time(amd_data, 'datetime')
	    amd_data['datetime'] = amd_data['datetime'].map(lambda x : pd.to_datetime(x))
	    amd_data.head()
	elif target_place == 3:
	    # 甲府アメダスのデータを使って予測する, amd_49142
	    # 各amdidはamd_masterに記載されている
	    amd_data = pd.read_csv('data/raw_data/amd_49142.tsv', delimiter = '\t')
	    amd_data = set_time(amd_data, 'datetime')
	    amd_data['datetime'] = amd_data['datetime'].map(lambda x : pd.to_datetime(x))
	    amd_data.head()
	else:
	    raise ValueError("invalid input target_place_num")

	# モデル構築のためにデータを分割する

	# 日射量の欠損値を一つ前の値で置換/output_data
	amd_data['max_tp'] = amd_data['max_tp'].fillna(method='bfill')

	# 学習に必要なデータ
	# 144ステップ分のデータで次ステップを予測する
	train_x_s_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2012-01-01 00:10')].index[0]
	train_x_e_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2015-12-31 23:40')].index[0]
	train_y_s_idx = train_x_s_idx + chunk_size
	train_y_e_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2015-12-31 23:50')].index[0]

	output_tp = amd_data[['max_tp']][train_y_s_idx:train_y_e_idx+1]
	input_tp = amd_data[['max_tp']][train_x_s_idx:train_x_e_idx+1]

	# 予測に必要なデータ
	# 144ステップ分のデータで次の予測をする
	test_y_s_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2015-12-31 00:00')].index[0]
	test_y_e_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2017-3-31 23:50')].index[0]
	test_x_s_idx = test_y_s_idx - chunk_size
	test_x_e_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2017-3-31 23:40')].index[0]

	test_input_tp = amd_data[['max_tp']][test_x_s_idx:test_x_e_idx+1]
	test_output_tp = amd_data[['max_tp']][test_y_s_idx:test_y_e_idx+1]

	# rnnに突っ込むための準備

	# rnnに突っ込むためにmin-max正規化しておく
	normalized_input = (input_tp - input_tp.min()) / (input_tp.max() - input_tp.min())
	normalized_input = np.array(normalized_input)
	normalized_output = (output_tp - output_tp.min()) / (output_tp.max() - output_tp.min())
	normalized_output = np.array(normalized_output)

	# 時系列データのリストにする
	input_list = get_chunked_data(normalized_input, chunk_size)

	# testデータの入力を用意
	normalized_test_input = (test_input_tp - test_input_tp.min()) / (test_input_tp.max() - test_input_tp.min())
	normalized_test_input = np.array(normalized_test_input)
	test_input_list = get_chunked_data(normalized_test_input, chunk_size)

	# denormalize用
	output_max = float(input_tp.max())
	output_min = float(input_tp.min())

	# RNNに突っ込むためにデータを整形
	X = np.array(input_list).reshape(len(input_list), chunk_size, input_list.shape[2])
	Y = np.array(normalized_output).reshape(len(input_list), 1)
	X_predict = np.array(test_input_list).reshape(len(test_input_list), chunk_size, test_input_list.shape[2])


	tp_lstm = simpleLSTM.simpleLSTM(X, Y, model_name = model_name)

	print("---- start prediction----")
	batch_size = 10000
	n_batch = len(test_input_list) // batch_size
	processed_predict = np.array([])
	for i in range(n_batch+1):
	    s_idx = i * batch_size
	    e_idx = (i+1) * batch_size
	    if e_idx > len(test_input_list):
	        e_idx = len(test_input_list)
	    print("---- predict " + str(s_idx) + " ~ " + str(e_idx)+ " ----")
	    predict = tp_lstm.predict(test_input_list[s_idx:e_idx], model_name)
	    predict = np.array(predict).reshape(len(predict[0]))
	    tmp_predict = denormalize_array(predict, output_min, output_max)
	    processed_predict = np.r_[processed_predict, tmp_predict]


	# In[132]:
	print("MESE : ", calc_mae(processed_predict, np.array(test_output_tp)))
	print("---- predicted tp ---- ")
	print(processed_predict[0:100])
	print()
	print("----- actual value -----")
	print(np.array(test_output_tp).reshape(len(test_output_tp))[0:100])
	print("---- finish prediction -----")

if __name__ == '__main__':
	main()
