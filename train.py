
# coding: utf-8

# # **浮島発電所の発電量を予測してみる**

# データ加工・処理・分析モジュール
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import simpleLSTM

from preprocess import *

def main():

	# パーサーを作る
	parser = argparse.ArgumentParser(
			prog='training lstm ', # プログラム名
            usage='target_place is required value', # プログラムの利用方法
            description='you can adjust place to predict, epochs, batch_size, loss_scale, chunk_size of lstm', # 引数のヘルプの前に表示
            epilog='end', # 引数のヘルプの後で表示
            add_help=True, # -h/–help オプションの追加
            )
	# 引数の追加
	parser.add_argument('-p', '--target_place', help='select place', type=int, required = True)
	parser.add_argument('-e', '--epochs', help='select epochs', type=int, required = False)
	parser.add_argument('-b', '--batch_size', help='select batch_size', type=int, required = False)
	parser.add_argument('-s', '--loss_scale', help='select place', type=int, required = False)
	parser.add_argument('-c', '--chunk_size', help='select chunk_size', type=int, required = False)
	# 入力引数
	args = parser.parse_args()
	target_place = args.target_place
	epochs = 100
	if args.epochs:
		epochs = args.epochs
	batch_size = 150
	if args.batch_size:
		batch_size = args.batch_size
	loss_scale = 1.0
	if args.loss_scale:
		loss_scale = args.loss_scale
	chunk_size = 144
	if args.chunk_size:
		chunk_size = args.chunk_size

	model_name = "model_"+str(target_place)
	print("---- trainning " + model_name + " ----")

	print("target_place : " + str(target_place) + ", epochs : " + str(epochs) + 
			", batch_size : " + str(batch_size) + ", loss_scale : " + str(loss_scale) + 
			", chunk_size :" + str(chunk_size))

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

	print("---- start training ----")
	model_01 = simpleLSTM.simpleLSTM(X, Y, epochs = epochs, batch_size = batch_size, loss_scale = loss_scale, model_name = model_name)

	model_01.train()
	print("---- finish training ----")

if __name__ == '__main__':
	main()