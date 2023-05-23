import keras
import onnx
import keras2onnx

# 加载 Keras 模型
model = keras.models.load_model('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/densenet121-imagenet_origin/H5/densenet121-imagenet_origin_origin0-WS0.h5')
#model = keras.models.load_model('/home/lemon_proj/IST_LEMON/origin_model/lenet5-fashion-mnist_origin.h5')
# model.summary()
# 将 Keras 模型转换为 ONNX 模型
onnx_model = keras2onnx.convert_keras(model, model.name)

# 保存 ONNX 模型
onnx.save(onnx_model, '/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/densenet121-imagenet_origin/ONNX/densenet121-imagenet_origin_origin0-WS0.onnx')
#onnx.save(onnx_model, '/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-fashion-mnist_origin/ONNX/lenet5-fashion-mnist_origin.onnx')


# # 检查 ONNX 模型
# onnx_model = onnx.load('/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/lenet5-mnist_origin_origin0-LA0.onnx')
# onnx.checker.check_model(onnx_model)