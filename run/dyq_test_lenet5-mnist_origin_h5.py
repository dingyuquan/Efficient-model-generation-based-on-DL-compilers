# import keras
# import onnx
# import torch
# from keras.datasets import mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# x_test = x_test.astype('float32')
# x_test /= 255
# y_test = keras.utils.to_categorical(y_test, 10)
#
# #model = keras.models.load_model('/home/lemon_proj/IST_LEMON/origin_model/lenet5-mnist_origin.h5')
# #model = keras.models.load_model('/home/lemon_proj/IST_LEMON/lemon_outputs/hll/lenet5-mnist_origin_origin0-CR0.h5')
# #model = keras.models.load_model('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin_origin0-ConvReplace0.h5')
# #model = torch.load('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin.pth')
# model = keras.models.load_model('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin/H5/lenet5-mnist_origin_origin0.h5')
#
# model.summary()
# #print(model)
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# score = model.evaluate(x_test, y_test, verbose=0)
#
# print(score)


import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from keras.datasets import mnist
from keras.models import load_model
from PIL import Image


# Load the pre-trained Keras model
keras_model_path = '/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin/H5/lenet5-mnist_origin_origin0-LA0.h5'
keras_model = load_model(keras_model_path)

# # Load the pre-trained PyTorch model
# pytorch_model_path = '/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin/PTH/lenet5-mnist_origin.pt'
# pytorch_model = torch.jit.load(pytorch_model_path)

# Load the pre-trained ONNX model
onnx_model_path = '/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin/ONNX/lenet5-mnist_origin_origin0-LA0.onnx'
sess = ort.InferenceSession(onnx_model_path)

# Load the test data
(x_test, y_test), _ = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255

# Evaluate the Keras model on the test data
keras_preds = keras_model.predict(x_test)
keras_preds = np.argmax(keras_preds, axis=1)

# Evaluate the PyTorch model on the test data
# pytorch_preds = []
# for i in range(x_test.shape[0]):
#     img_tensor = torch.from_numpy(x_test[i]).unsqueeze(0).float()
#     pytorch_pred = np.argmax(pytorch_model(img_tensor).detach().numpy())
#     pytorch_preds.append(pytorch_pred)
# pytorch_preds = np.array(pytorch_preds)

# Evaluate the ONNX model on the test data
onnx_preds = []
for i in range(x_test.shape[0]):
    onnx_input = {sess.get_inputs()[0].name: x_test[i].reshape(1, 28, 28, 1)}
    onnx_pred = np.argmax(sess.run(None, onnx_input)[0])
    onnx_preds.append(onnx_pred)
onnx_preds = np.array(onnx_preds)

# Compare the results
# if np.array_equal(keras_preds, pytorch_preds) and np.array_equal(pytorch_preds, onnx_preds):
#     print("All models produced the same predictions.")
# else:
#     if not np.array_equal(keras_preds, pytorch_preds):
#         print("Keras and PyTorch models produced different predictions.")
#         print("Keras predictions:", keras_preds)
#         print("PyTorch predictions:", pytorch_preds)
#     if not np.array_equal(pytorch_preds, onnx_preds):
#         print("PyTorch and ONNX models produced different predictions.")
#         print("PyTorch predictions:", pytorch_preds)
#         print("ONNX predictions:", onnx_preds)
#     if not np.array_equal(keras_preds, onnx_preds):
#         print("Keras and ONNX models produced different predictions.")
#         print("Keras predictions:", keras_preds)
#         print("ONNX predictions:", onnx_preds)



if np.array_equal(keras_preds, onnx_preds):
    print("All models produced the same predictions.")
else:
    if not np.array_equal(keras_preds, onnx_preds):
        print("Keras and ONNX models produced different predictions.")
        print("Keras predictions:", keras_preds)
        print("ONNX predictions:", onnx_preds)

print(keras_preds, onnx_preds)