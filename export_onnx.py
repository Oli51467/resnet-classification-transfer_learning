import onnx
import torch
from onnxsim import simplify

from model import resnet34

INPUT_DICT = './resnet_transfer_learning/weight/resNet34.pth'
OUT_ONNX = './resnet_transfer_learning/weight/best.onnx'

x = torch.randn(1, 3, 224, 224)
input_names = ["input"]
out_names = ["output"]

model = resnet34(3)
model.load_state_dict(torch.load(INPUT_DICT, map_location=torch.device('cpu')))
#model = torch.load(INPUT_DICT, map_location=torch.device('cpu'))

model.eval()

torch.onnx._export(model, x, OUT_ONNX, export_params=True, training=False, input_names=input_names, output_names=out_names)
onnx_model = onnx.load('./resnet_transfer_learning/weight/best.onnx')
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, './resnet_transfer_learning/weight/best-sim.onnx')
