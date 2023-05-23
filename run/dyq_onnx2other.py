import onnx
from onnx2torch import convert
import torch
import tensorflow as tf

#onnx
#onnx_model = onnx.load('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-fashion-mnist_origin/ONNX/lenet5-fashion-mnist_origin_origin0-MLA0.onnx')
onnx_model = onnx.load('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/densenet121-imagenet_origin/ONNX/densenet121-imagenet_origin_origin0-WS0.onnx')

onnx.checker.check_model(onnx_model)

# #tensorflow
# from onnx_tf.backend import prepare
#
# tf_model = prepare(onnx_model)


#pytorch
pytorch_model = convert(onnx_model)

torch.save(pytorch_model, '/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/densenet121-imagenet_origin/PTH/densenet121-imagenet_origin_origin0-WS0.pth') #保存网络结构
#new_model = torch.load('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin/PTH/lenet5-mnist_origin_origin0-LA0.pth')


#print(new_model)



