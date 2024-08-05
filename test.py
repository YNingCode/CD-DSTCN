import torch

def windows(input):
    windows_num = 5
    batch, node, len = input.size()
    patch_size = len // windows_num
    input = input[:,:, :patch_size*windows_num]
    input = input.reshape(batch, -1, node, patch_size)
    return input


# 生成一个示例tensor
tensor = torch.randn(203, 116, 220)
out = windows(tensor)
print(out.shape)