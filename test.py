import torch
import torch.nn as nn
from torch.ao.quantization import quantize

from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as tf


def calc_weight_loss(weights, mean:float, std:float, step:int, scale:float=1.96):
    sep = 2 * scale * std / step
    qp = torch.linspace(- scale * std, scale * std, step + 1)
    weights = weights % sep
    weights[weights > (sep / 2)] = weights[weights > (sep / 2)] - sep
    weights = weights.view(-1, 1)
    qp = qp.view(1, step+1)

    gap = torch.remainder(weights - qp, sep)
    return gap


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        x = self.dequant(x)
        return x


def main():
    transform = tf.Compose([
        tf.PILToTensor()
    ])
    dataset = MNIST('/home/regularized_quantization/data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
        
    model_fp32 = M()

    model_fp32.eval()

    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

    input_fp32 = torch.tensor([-32,32,32,
                                32,32,32,
                                64,96,256], dtype=torch.float32).view(1,1,3,3)

    model_fp32_prepared(input_fp32)

    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    res = model_int8(input_fp32)



if __name__ == '__main__':
    main()

