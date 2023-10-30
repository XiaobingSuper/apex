import torch

import apex

import upsample_nearest2d_cuda

class UserFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_size, scale_factor):
        output = upsample_nearest2d_cuda.forward(x, output_size, scale_factor)
        
        ctx.input_size = x.size()
        ctx.output_size = output.size()[-2:]
        ctx.scale_factor = scale_factor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_size = ctx.input_size
        output_size = ctx.output_size
        scale_factor = ctx.scale_factor
        print("cccccccccccccccccc")
        grad_input = upsample_nearest2d_cuda.backward(grad_output, output_size, input_size, scale_factor)
        return grad_input, None, None

torch._C._nn.upsample_nearest2d = UserFunction.apply

x = torch.randn(1, 3, 4, 4, dtype=torch.bfloat16).cuda().requires_grad_(True)

scale_factor = (2.0, 2.)
output_shape = (8, 8)

y = torch._C._nn.upsample_nearest2d(x, None, scale_factor)

y.sum().backward()

#y = upsample_nearest2d_cuda.forward(x, output_shape, scale_factor)

