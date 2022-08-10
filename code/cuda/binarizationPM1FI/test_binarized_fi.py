import torch
import binarizePM1FI
import binarizePM1

# TODO: create better and comprehensive test cases
# e.g. calculating probabilities

tensor = torch.rand(size=(1,1,5,5), dtype=torch.float).cuda()
tensor *= 100
tensor -= torch.mean(tensor)
binarizePM1.binarize(tensor)
print("random tensor", tensor)

binarizePM1FI.binarizeFI(tensor, 0.3, 0.3)
print("random tensor", tensor)
