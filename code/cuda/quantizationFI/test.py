import torch
import quantization
import quantizationFI

tensor = torch.rand(size=(2,2,3,3), dtype=torch.float).cuda()
tensor *= 100
tensor -= torch.mean(tensor)
# print("random tensor", tensor)
a = tensor
bits = 8
a = quantization.quantize(a, a.min().item(), tensor.max().item(), bits, 1)
print("unsigned quantization", a)
a = quantizationFI.bfi_8bit(a, 0.1, 0.1, bits)
print("bitflips injected", a)
print("------------------------------------")

# b = qtize.shift_bfi_4bit(a, 0.1, 0.1)
# print(b)
