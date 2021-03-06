# データ加工・処理・分析モジュール
import numpy as np
import tensorflow as tf
import random

class two_layer:
    def __init__(self, X, Y, X_val, Y_val, epochs = 100, hidden_1 = 100, hidden_2 = 100, batch_size = 100, model_name = "test_model", activation = "sigmoid"):
        # 学習データと検証用データに分けておく
        self.X = X # 入力
        self.Y = Y # 教師
        self.X_val = X_val # 検証用
        self.Y_val = Y_val #検証用
        
        '''
        諸変数の設定
        '''
        self.input_layer_size = self.X.shape[1] #入力層の数
        self.hidden1_layer_size = hidden_1 # 隠れ層の数、適当
        self.hidden2_layer_size = hidden_2 # 隠れ層の数、適当
        self.output_layer_size = self.Y.shape[1] #出力層の数
        self.batch_size = batch_size #バッチサイズ
        self.learning_rate = 0.01 # 学習率 適当
        self.epochs = epochs #エポック数
        self.activation = activation
        
        # 学習データの保存
        self.model_name = str(model_name)
        
        
    def shuffle(self):
        '''
        ミニバッチかする際にランダムにシャッフル
        '''
        idx = [i for i in range(self.X.shape[0])]
        np.random.shuffle(idx)
        xs = np.array([[y for y in list(self.X[r])] for r in idx])
        ys = np.array([self.Y[r] for r in idx])
        return xs, ys
        
    def inference(self, input_ph):
        '''
        グラフの構築
        '''
        
        # 重みとバイアスの初期化
        hidden1_w = tf.Variable(tf.random_normal([self.input_layer_size, self.hidden1_layer_size], stddev=0.01), name='hidden1_w')
        hidden1_b = tf.Variable(tf.random_normal([self.hidden1_layer_size]), name='hidden1_b')
        hidden2_w = tf.Variable(tf.random_normal([self.hidden1_layer_size, self.hidden2_layer_size], stddev=0.01), name='hidden2_w')
        hidden2_b = tf.Variable(tf.random_normal([self.hidden2_layer_size]), name='hidden2_b')
        output_w = tf.Variable(tf.random_normal([self.hidden2_layer_size, self.output_layer_size], stddev=0.01), name='output_w')
        output_b = tf.Variable(tf.random_normal([self.output_layer_size]), name='output_b')
        
        # 計算
        if  self.activation == "sigmoid":
            hidden1 = tf.sigmoid(tf.matmul(input_ph, hidden1_w) + hidden1_b)
            hidden2 = tf.sigmoid(tf.matmul(hidden1, hidden2_w) + hidden2_b)
        elif  self.activation == "relu":
            hidden1 = tf.nn.relu(tf.matmul(input_ph, hidden1_w) + hidden1_b)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden2_w) + hidden2_b)
        elif self.activation == "tanh":
            hidden1 = tf.tanh(tf.matmul(input_ph, hidden1_w) + hidden1_b)
            hidden2 = tf.tanh(tf.matmul(hidden1, hidden2_w) + hidden2_b)

        output = tf.matmul(hidden2, output_w) + output_b

        weights = [hidden1_w, hidden2_w, output_w, hidden1_b, hidden2_b, output_b]
        
        return output, weights
        
    def loss(self, output_ph, actual_ph):
        '''
        MSEを使用
        '''
        cost = tf.reduce_mean(tf.square((output_ph - actual_ph)))
        tf.summary.scalar('loss', cost)
        return cost
    
    def training(self, cost):
        '''
        adamを仕様beta1, beta2は元論文の推奨値を仕様
        '''
        with tf.name_scope("training") as scope:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(cost)
            return optimizer
    
    def train(self):
        '''
        学習
        '''
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)
        n_batch = self.X.shape[0] // self.batch_size

        # early stop用
        validation_losses = [9999, 9999, 9999]
        
        with tf.Graph().as_default():
            # 変数の用意
            input_ph = tf.placeholder('float', [None, self.input_layer_size], name='input')
            actual_ph = tf.placeholder('float', [None, self.output_layer_size], name='actual_value')

            prediction, weights = self.inference(input_ph)
            cost = self.loss(prediction, actual_ph)
            optimizer = self.training(cost)

            # TensorBoardで可視化する
            summary = tf.summary.merge_all()
            # 初期化
            init = tf.global_variables_initializer()
                
            # ここから学習
            with tf.Session() as sess:
                # 学習したモデルも保存しておく
                saver = tf.train.Saver()
                summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
                sess.run(init)

                for epoch in range(self.epochs):
                    X_, Y_ = self.shuffle()
                    for i in range(n_batch):
                        start = i * self.batch_size
                        end = start + self.batch_size
                        inputs  = X_[start:end]
                        actuals = Y_[start:end]
                        train_dict = {
                            input_ph:      inputs,
                            actual_ph:     actuals,
                        }
                    
                    sess.run(optimizer, feed_dict=train_dict)

                    if (epoch) % (self.epochs//50) == 0:
                        val_dict = {
                            input_ph:      self.X_val,
                            actual_ph:     self.Y_val,
                        }
                        summary_str, train_loss = sess.run([summary, cost], feed_dict=val_dict)
                        print("train#%d, validation loss: %e" % (epoch, train_loss))
                        summary_writer.add_summary(summary_str, epoch)

                        if validation_losses[-1] < train_loss and validation_losses[-2] < train_loss and validation_losses[-3] < train_loss:
                            print("do early stopping")
                            break

                        validation_losses.append(train_loss)

                    datas = sess.run(weights)
                    saver.save(sess,  "./data/model/" + str(self.model_name) + "/" + str(self.model_name) + ".ckpt")
                    
                # datas = sess.run(weights)
                # saver.save(sess, "./data/model/" + str(self.model_name) + "/" + str(self.model_name) + ".ckpt")
                
    def predict(self, X_predict, model_name = "test_model"):
        '''
        予測期間に該当するデータから予測
        '''
        # 予測に使う変数の用意
        tf.reset_default_graph()
        input_ph = tf.placeholder("float", [None, self.input_layer_size], name='input')
        prediction, weights = self.inference(input_ph)
        pre_dict = {
            input_ph: X_predict,
        }
        
        # 初期化
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # 保存したモデルをロード
            saver = tf.train.Saver()
            saver.restore(sess,  "./data/model/" + str(self.model_name) + "/" + str(self.model_name) + ".ckpt")

            # ロードしたモデルを使って予測結果を計算
            expected_output = sess.run([prediction], feed_dict=pre_dict)

        return expected_output
    
    def features(self, input_ph):
        '''
        Auto_Encoder用
        '''
        
        # 重みとバイアスの初期化
        hidden_w = tf.Variable(tf.truncated_normal([self.input_layer_size, self.hidden_layer_size], stddev=0.01), name='hidden_w')
        hidden_b = tf.Variable(tf.truncated_normal([self.hidden_layer_size]), name='hidden_b')
        output_w = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.output_layer_size], stddev=0.01), name='output_w')
        output_b = tf.Variable(tf.truncated_normal([self.output_layer_size]), name='output_b')
        
        # 計算
        hidden = tf.sigmoid(tf.matmul(input_ph, hidden_w) + hidden_b)
        output = tf.sigmoid(tf.matmul(hidden, output_w) + output_b)
        
        weights = [hidden_w, output_w, hidden_w, hidden_b]
        
        return hidden, weights
    
    def get_features(self, X, model_name = "test_model"):
        '''
        圧縮された特徴量を得る
        '''
        # 予測に使う変数の用意
        tf.reset_default_graph()
        input_ph = tf.placeholder("float", [None, self.input_layer_size], name='input')
        prediction, weights = self.features(input_ph)
        pre_dict = {
            input_ph: X,
        }
        
        # 初期化
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # 保存したモデルをロード
            saver = tf.train.Saver()
            saver.restore(sess,  "./data/model/" + str(self.model_name) + "/" + str(self.model_name) + ".ckpt")

            # ロードしたモデルを使って予測結果を計算
            expected_output = sess.run([prediction], feed_dict=pre_dict)


        return expected_output