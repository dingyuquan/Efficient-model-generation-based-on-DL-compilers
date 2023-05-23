# import numpy as np
# import onnx
# import onnxruntime as ort
# import tvm
# from tvm import relay
# from tvm.contrib import graph_executor
# from tvm.relay import testing
# import torch
# import torchvision.transforms as transforms
# from keras.datasets import mnist
# from keras.models import load_model
# from PIL import Image
#
#
# # Load the pre-trained Keras model
# keras_model = load_model('lenet5-mnist.h5')
#
# # Load the pre-trained PyTorch model
# pytorch_model_path = 'lenet5-mnist.pt'
# pytorch_model = torch.jit.load(pytorch_model_path)
#
# # Load the pre-trained ONNX model
# onnx_model_path = 'lenet5-mnist.onnx'
# onnx_model = onnx.load(onnx_model_path)
# onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
#
# # Compile the Keras model using TVM
# input_name = 'input'
# shape_list = [(input_name, (1, 28, 28, 1))]
# keras_model = relay.frontend.from_keras(keras_model, shape_list)
# mod, params = relay.frontend.from_keras(keras_model, shape_list)
# target = 'llvm'
# with tvm.transform.PassContext(opt_level=3):
#     graph, lib, params = relay.build(mod, target, params=params)
# module = graph_executor.GraphModule(lib['default'](tvm.cpu(0)))
# module.set_input(**params)
#
# # Compile the PyTorch model using TVM
# input_name = 'input'
# shape_list = [(input_name, (1, 1, 28, 28))]
# input_shapes = [(input_name, (1, 1, 28, 28))]
# pytorch_model = testing.create_workload(pytorch_model, input_shapes)
# mod, params = relay.frontend.from_pytorch(pytorch_model, input_shapes)
# target = 'llvm'
# with tvm.transform.PassContext(opt_level=3):
#     graph, lib, params = relay.build(mod, target, params=params)
# module2 = graph_executor.GraphModule(lib['default'](tvm.cpu(0)))
# module2.set_input(**params)
#
# # Compile the ONNX model using TVM
# input_name = 'input'
# shape_list = [(input_name, (1, 28, 28, 1))]
# onnx_model = onnxruntime.transformers.onnx_model_bake(onnx_model)
# mod, params = relay.frontend.from_onnx(onnx_model, shape_list)
# target = 'llvm'
# with tvm.transform.PassContext(opt_level=3):
#     graph, lib, params = relay.build(mod, target, params=params)
# module3 = graph_executor.GraphModule(lib['default'](tvm.cpu(0)))
# module3.set_input(**params)
#
# # Load the test data
# (x_test, y_test), _ = mnist.load_data()
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# x_test = x_test.astype('float32') / 255
#
# # Evaluate the Keras model using TVM on the test data
# keras_preds = []
# for i in range(x_test.shape[0]):
#     tvm_input = tvm.nd.array(x_test[i].reshape(1, 28, 28, 1))
#     module.set_input(input_name, tvm_input)
#     module.run()
#     tvm_pred = module.get_output(0).asnumpy().argmax()
#     keras_preds.append(tvm_pred)
# keras_preds = np.array(keras_preds)
#
# # Evaluate the PyTorch model using TVM on the test data
# pytorch_preds = []
# for i in range(x_test.shape[0]):
#     tvm_input = tvm.nd.array(x_test[i].reshape(1, 1, 28, 28))
#     module2.set_input(input_name, tvm_input)
#     module2.run()
#     tvm_pred = module2.get_output(0).asnumpy().argmax()
#     pytorch_preds.append(tvm_pred)
# pytorch_preds = np.array(pytorch_preds)
#
# # Evaluate the ONNX model using TVM on the test data
# onnx_preds = []
# for i in range(x_test.shape[0]):
#     tvm_input = tvm.nd.array(x_test[i].reshape(1, 28, 28, 1))
#     module3.set_input(input_name, tvm_input)
#     module3.run()
#     tvm_pred = module3.get_output(0).asnumpy().argmax()
#     onnx_preds.append(tvm_pred)
# onnx_preds = np.array(onnx_preds)
#
# # Compare the results
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
#     if not np.array_equal(onnx_preds, keras_preds):
#         print("ONNX and Keras models produced different predictions.")
#         print("ONNX predictions:", onnx_preds)
#         print("Keras predictions:", keras_preds)

import keras
import onnx
import keras2onnx

# 加载 Keras 模型
model = keras.models.load_model('/home/lemon_proj/IST_LEMON/origin_model/lenet5-mnist_origin.h5')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('/home/lemon_proj/IST_LEMON/origin_model/lenet5-mnist_origin.h5')

model.summary()
if model.optimizer:
    print("yes")
else:
    print("no")