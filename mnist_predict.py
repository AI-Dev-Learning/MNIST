# ******************************************************************************************************
# 打印Tenserflow 版本
import tensorflow as tf
print(tf.__version__)

# 打印keras版本
import keras
print(keras.__version__)


import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
np.random.seed(10)

# ******************************************************************************************************
# 匯入資料
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print('train data= ',len(x_train_image))
print('test data=', len(x_test_image))


# ******************************************************************************************************
# 畫一張圖片
import matplotlib.pyplot as plt

def plot_image(image):
  fig = plt.gcf()
  fig.set_size_inches(2,2)
  plt.imshow(image,cmap='binary')
  plt.show()
plot_image(x_train_image[100])
y_train_label[0]

# ******************************************************************************************************
# 畫多張圖片
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
print(x_Train.shape)
print(x_Test.shape)
print(x_Train[0])



# 由於是圖片最大的是255，所以全部除以255

x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255

print(x_Train_normalize[0])





# 查看原本的 label 型態
# 他是0~9的數字
y_train_label[:5]

y_TrainOneHot=to_categorical(y_train_label)
y_TestOneHot=to_categorical(y_test_label)



# ******************************************************************************************************
# 加載模型以及預測
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model


# loaded_model = load_model('my_model.h5')
loaded_model = keras.models.load_model('my_model.keras')

# 測試模型
loaded_model.summary()
scores = loaded_model.evaluate(x_Test_normalize, y_TestOneHot)
print("Loaded model accuracy:", scores[1])



prediction=loaded_model.predict(x_Test)
prediction = np.argmax(prediction, axis=1)

# prediction

plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=340)

# ******************************************************************************************************
# 顯示混淆矩陣
# 檢查模型性能：
# 對角線上的值表示模型預測正確的次數。
# 非對角線的值表示錯誤分類的次數。
import pandas as pd
confusion_matrix  = pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['prediction'])
print(confusion_matrix)

df = pd.DataFrame({'label':y_test_label,'predict':prediction})
df[:10]
print(df.head(10))




filtered_df  = df[(df.label==5)&(df.predict==3)]
print(filtered_df)

plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=340,num=1)

