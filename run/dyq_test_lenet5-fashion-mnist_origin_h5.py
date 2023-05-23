import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# 加载 Fashion MNIST 数据集
(_, _), (test_images, test_labels) = fashion_mnist.load_data()

# 将数据集的像素值缩放到 [0, 1] 范围内，并转换成浮点型
test_images = test_images / 255.0

# 加载 LeNet-5 模型
model = tf.keras.models.load_model('/home/lemon_proj/IST_LEMON/origin_model/lenet5-fashion-mnist_origin.h5')

# 在测试集上进行预测
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
score = model.evaluate(test_images[..., tf.newaxis], test_labels)

print(score)