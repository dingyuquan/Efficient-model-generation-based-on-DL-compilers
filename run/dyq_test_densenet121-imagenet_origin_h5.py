import numpy as np
from keras.utils import to_categorical
from keras_applications.densenet import preprocess_input

#加载.npz文件
data = np.load('/home/lemon_proj/lyh/dataset/imagenet-val-1500.npz',allow_pickle=True)

# 从NpzFile对象中读取数据集数组或矩阵
x_test = data['x_test']
y_test = data['y_test']
# y_test = to_categorical(y_test, num_classes=1000).astype('float32')


from keras.models import load_model

# 加载之前训练好的模型
model = load_model('/home/lemon_proj/IST_LEMON/origin_model/densenet121-imagenet_origin.h5')
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=1)

print(score)
