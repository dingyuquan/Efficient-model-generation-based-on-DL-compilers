import keras
from keras.datasets import cifar10
from keras.utils import to_categorical

# 加载测试集
(x_test, y_test), _ = cifar10.load_data()
y_test = to_categorical(y_test)

from keras.models import load_model

# 加载之前训练好的模型
model = load_model('/home/lemon_proj/IST_LEMON/origin_model/alexnet-cifar10_origin.h5')

# 将图像转换为浮点数并归一化
x_test = x_test.astype('float32')
x_test /= 255

# 计算测试集损失值和准确率
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=0)

print(score)
# test_loss = score[0]
# test_acc = score[1]
#
# print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(test_loss, test_acc*100))