import torch
import quantization

tensor = torch.rand(size=(2,2,3,3), dtype=torch.float).cuda()
tensor *= 100
tensor -= torch.mean(tensor)
print("random tensor", tensor)
a = quantization.quantize(tensor, tensor.min().item(), tensor.max().item(), 8, 0)
print("signed quantization", a)
a = quantization.quantize(a, a.min().item(), tensor.max().item(), 8, 1)
print("unsigned quantization", a)
a = quantization.quantize(a, a.min().item(), tensor.max().item(), 8, 0)
print("signed quantization", a)
print("------------------------------------")

# b = qtize.shift_bfi_4bit(a, 0.1, 0.1)
# print(b)
