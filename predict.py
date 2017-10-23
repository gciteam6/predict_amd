
# coding: utf-8

# # **浮島発電所の発電量を予測してみる**
# データ加工・処理・分析モジュール
import numpy as np
import pandas as pd
import argparse
import simpleLSTM
from preprocess import *

def main():
	
	# In[109]:
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
	# 入力引数
	args = parser.parse_args()
	target_place = args.target_place
	chunk_size = args.chunk_size
	model_name = "model_"+str(target_place)
	print("---- predict by  " + model_name + " ----")
	print("target_place : " + str(target_place) + ", chunk_size : " + str(chunk_size))

	print("---- load output data ----")
	# 10分単位の発電量のデータを取ってくる
	output_data = pd.read_csv('data/raw_data/train_kwh.tsv', delimiter = '\t')

	# datetimeの行をpd.Timestampのインスタンスに変更
	output_data = set_time(output_data, 'datetime')
	output_data['datetime'] = output_data['datetime'].map(lambda x : pd.to_datetime(x))

	print("---- load input data ----")
	# アメダスデータの読み込み
	if target_place == 1 or target_place == 2:
	    # 横浜アメダスのデータを使って予測する, amd_46106
	    # 各amdidはamd_masterに記載されている
	    amd_data = pd.read_csv('data/raw_data/amd_46106.tsv', delimiter = '\t')
	    amd_data = set_time(amd_data, 'datetime')
	    amd_data['datetime'] = amd_data['datetime'].map(lambda x : pd.to_datetime(x))
	elif target_place == 3:
	    # 甲府アメダスのデータを使って予測する, amd_49142
	    # 各amdidはamd_masterに記載されている
	    amd_data = pd.read_csv('data/raw_data/amd_49142.tsv', delimiter = '\t')
	    amd_data = set_time(amd_data, 'datetime')
	    amd_data['datetime'] = amd_data['datetime'].map(lambda x : pd.to_datetime(x))
	else:
	    raise ValueError("invalid input target_place_num")


	#学習データと予測データに分割
	train_amd_data, train_output_data, test_amd_data = get_dataset(amd_data, output_data['SOLA0'+str(target_place)], chunk_size)

	# rnnに突っ込むためにデータを整形
	X, Y, X_predict, output_min, output_max = prepare_rnn(train_amd_data, test_amd_data, train_output_data, chunk_size)

	model_01 = simpleLSTM.simpleLSTM(X, Y)

	print("---- calc training error----")
	# train_error
	batch_size = 10000
	n_batch = len(X[0:10000]) // batch_size
	processed_predict = np.array([])
	for i in range(n_batch+1):

	    s_idx = i * batch_size
	    e_idx = (i+1) * batch_size
	    if e_idx > len(X[0:10000]):
	        e_idx = len(X[0:10000])

	    predict = model_01.predict(X[s_idx:e_idx], model_name)

	    predict = np.array(predict).reshape(len(predict[0]))
	    tmp_predict = denormalize_array(predict, output_min, output_max)
	    processed_predict = np.r_[processed_predict, tmp_predict]

	print("actual trainning error : ", np.abs(train_output_data[0:10000] - processed_predict).mean())

	print("calculated_output : ", processed_predict[0:100])

	print("---- start predicttion ----")
	# prediction
	batch_size = 10000
	n_batch = len(X_predict) // batch_size
	processed_predict = np.array([])
	for i in range(n_batch+1):

	    s_idx = i * batch_size
	    e_idx = (i+1) * batch_size
	    if e_idx > len(X_predict):
	        e_idx = len(X_predict)

	    predict = model_01.predict(X_predict[s_idx:e_idx], model_name)

	    predict = np.array(predict).reshape(len(predict[0]))
	    tmp_predict = denormalize_array(predict, output_min, output_max)
	    processed_predict = np.r_[processed_predict, tmp_predict]

	print("expected_output : ", processed_predict[0:100])

	# 2016/01/01 00:00 ~ 2017/3/31 23:50の予測データを書き出す
	s_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2016/01/01 00:00')].index[0]
	e_idx = amd_data[amd_data['datetime'] == pd.to_datetime('2017/3/31 23:50')].index[0]
	predict_data = pd.DataFrame({"datetime":amd_data['datetime'][s_idx:e_idx+1], "expected_output":processed_predict})
	predict_data.index = np.arange(len(predict_data))
	predict_data.to_csv('data/predicted_data/predict_SOLA0'+str(target_place)+'.tsv', sep = '\t')

if __name__ == '__main__':
	main()