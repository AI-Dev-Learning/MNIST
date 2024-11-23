# ******************************************************************************************************
# 打印tenserflow 版本
import tensorflow as tf
print(tf.__version__)

# 打印keras 版本
import keras
print(keras.__version__)


import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
np.random.seed(10)

# ******************************************************************************************************
# 匯入MNIST資料
# image的部分
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print('train data= ',len(x_train_image))
print('test data=', len(x_test_image))

# train data = 60000
# test data = 10000

# ******************************************************************************************************

import matplotlib.pyplot as plt

# 畫一張圖片 x_train_image[100]
def plot_image(image):
  fig = plt.gcf()
  fig.set_size_inches(2,2)
  plt.imshow(image,cmap='binary')
  plt.show()

plot_image(x_train_image[100])
# 印出對印Label y_train_label[100]
print("Label for x_train_image[100]:", y_train_label[100])
# Label for x_train_image[100]: 5


# ******************************************************************************************************



import matplotlib.pyplot as plt

# 建立函數要來畫多圖的
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
  # 設定顯示圖形的大小
  fig = plt.gcf()
  fig.set_size_inches(12, 14)

  # 最多25張
  if num > 25: num = 25

  # 一張一張畫
  for i in range(0, num):

    # 建立子圖形5*5(五行五列)
    ax = plt.subplot(5, 5, i + 1)

    # 畫出子圖形
    ax.imshow(images[idx], cmap='binary')

    # 標題和label
    title = "label=" + str(labels[idx])

    # 如果有傳入預測結果也顯示
    if len(prediction) > 0:
      title += ",predict=" + str(prediction[idx])

    # 設定子圖形的標題大小
    ax.set_title(title, fontsize=10)

    # 設定不顯示刻度
    ax.set_xticks([]);
    ax.set_yticks([])
    idx += 1
  plt.show()

# ******************************************************************************************************

plot_images_labels_prediction(x_train_image, y_train_label, [], 0, 10)

print('x_test_image:',x_test_image.shape)
print('y_test_label:', y_test_label.shape)

plot_images_labels_prediction(x_test_image,y_test_label,[],0,25)

# ******************************************************************************************************
# 清理資料 data cleaning
# 代表 train image 總共有6萬張，每一張是28*28的圖片
# label 也有6萬個
# 所以要把二維的圖片矩陣先轉換成一維
# 這裡的784是因為 28*28
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Test=x_test_image.reshape(10000,784).astype('float32')

# 轉換後的資料型態，壓扁變成一維了
print(x_Train.shape) # (60000, 784)
print(x_Test.shape) # (60000, 784)
print(x_Train[0]) # 二維資料


# 影像標準化 Normailze
# 由於是圖片最大的是255，所以全部除以255
x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255
print(x_Train_normalize[0])



# label前處理 使用one-hot encoding
# 查看原本的 label 型態
# 他是0~9的數字
y_train_label[:5]
print(y_train_label[:5]) # [5 0 4 1 9]

y_TrainOneHot=to_categorical(y_train_label)
y_TestOneHot=to_categorical(y_test_label)

# 來看轉換好的
# 這個就是第一筆資料，他是數字5
print(y_TrainOneHot[:1]) # [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

# ******************************************************************************************************
# 建立模型 多元感知器 Multilayer perceptron
from keras.models import Sequential
from keras.layers import Dense

# 建立模型
model = Sequential()




# ************************************ 優化前 ************************************
# 建立輸入層和隱藏層
# model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
# # 定義隱藏層神經元個數256
# # 輸入為28*28=784 個float 數字
# # 使用 normal distribution 常態分布的亂數，初始化 weight權重 bias 偏差
# # 定義激活函數為 relu
#
#
# # 建立輸出層
# model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
# # 定義輸出層為10個 (數字0~9)
# # 也是使用常態分佈初始化
# # 定義激活函數是 softmax
# # 這裡建立的Dense 層，不用設定 input dim ，因為keras 會自動照上一層的256設定
#
# print(model.summary())


# Model: "sequential_4"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_9 (Dense)              (None, 256)               200960
# _________________________________________________________________
# dense_10 (Dense)             (None, 10)                2570
# =================================================================
# Total params: 203,530
# Trainable params: 203,530
# Non-trainable params: 0
# _________________________________________________________________
# None

# 從這個 summary 可以看出 這一個模型是兩層的模型
# 然後隱藏層有256個神經元
# 輸出層有10個神經元

# 另外是 param 參數
# 參數的計算方式第一個是 200960=256*784+256
# 另外一個是2570=256*10+10=2570
# 下面有一個全部訓練 total params=200960+2570=203530


# ************************************ 優化後 ************************************
# hidden layer增加為1000個神經源
# 加入 dropout 避免overfitting
# 建立多層感知模型包含2 個隱藏層
from keras.layers import Dropout

model= Sequential()

model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='softmax'))
model.add(Dropout(0.5))


model.add(Dense(units=1000,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))

print(model.summary())

# Model: "sequential_5"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_11 (Dense)             (None, 1000)              785000
# _________________________________________________________________
# dense_12 (Dense)             (None, 10)                10010
# =================================================================
# Total params: 795,010
# Trainable params: 795,010
# Non-trainable params: 0
# _________________________________________________________________
# None


# ******************************************************************************************************
# 開始訓練

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 模型訓練之前要用 compele 對模型進行設定
# loss 深度學習通常用 cross entropy 交叉嫡，訓練效果較好
# optimizer 設定訓練時依優化的方法，在深度學習使用 adam 最優化方法，最快收斂提高準確度
# metrics 設定評估模型的方式是 accuracy 準確率


# 開始訓練

train_history=model.fit(x=x_Train_normalize,y=y_TrainOneHot,
            validation_split=0.2,epochs=10,batch_size=200,verbose=2)
# x 是訓練資料
# y 是label 資料
# 設定參數 validation 切0.2起來驗證
# epoch=10 是訓練週期為10
# batch_size=200 每一批訓練200筆資料
# verbose =2 顯示訓練過程

# 所以以上的程式會執行10次
# 每一次執行200筆資料 ，總共訓練資料原本有60000*0.8=48000
# 48000/200=24 要跑240批次
# epoch 每一次訓練週期紀錄結果在 train_history 裡面



# 來把訓練過程畫出來
import matplotlib.pyplot as plt

def show_train_history(train_history,train,validation):

  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train history')
  plt.ylabel('train')
  plt.xlabel('epoch')

  # 設置圖例在左上角
  plt.legend(['train','validation'],loc='upper left')
  plt.show()

show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')


# ******************************************************************************************************
# 評估測試資料準確率

scores=model.evaluate(x_Test_normalize,y_TestOneHot)
print()
print('accuracy',scores[1])

# ******************************************************************************************************
# 保存完整模型（包括結構和權重）
model.save('my_model.keras')  # 保存為 HDF5 格式

prediction=model.predict(x_Test)
prediction = np.argmax(prediction, axis=1)

# ******************************************************************************************************