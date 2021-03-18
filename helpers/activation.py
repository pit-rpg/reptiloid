import torch
import torch.nn as nn

class ReptiloidReLU(nn.Module):
    def __init__(self):
        super().__init__()
        # self.device = device
        # self.min = torch.Tensor(0.0, device=device)
        # self.max = torch.Tensor(1.0, device=device)
        # self.min_mul = torch.Tensor(0.1, device=device)
        # self.max_mul = torch.Tensor(-0.75, device=device)

    def forward(self, x):
        return torch.where(x < 0.0, x * 0.1, torch.where(x > 1.0, x * -0.75, x))
        # return torch.where(x < self.min, x * self.min_mul, torch.where(x > self.max, x * self.max_mul, x))

class SReptiloidReLU(nn.Module):
    def __init__(self):
        super().__init__()
        # self.device = device
        # self.min = torch.Tensor(0.0, device=device)
        # self.max = torch.Tensor(1.0, device=device)
        # self.min_mul = torch.Tensor(0.1, device=device)
        # self.max_mul = torch.Tensor(-0.75, device=device)

    def forward(self, x):
        return torch.where(x < 0.0, x * -0.05, torch.where(x > 1.0, x * -0.05, x))
        # return torch.where(x < self.min, x * self.min_mul, torch.where(x > self.max, x * self.max_mul, x))

# class Linear(nn.Module):
#     def __init__(self, input_features, output_features, bias=True):
#         super(Linear, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features

#         # nn.Parameter is a special kind of Tensor, that will get
#         # automatically registered as Module's parameter once it's assigned
#         # as an attribute. Parameters and buffers need to be registered, or
#         # they won't appear in .parameters() (doesn't apply to buffers), and
#         # won't be converted when e.g. .cuda() is called. You can use
#         # .register_buffer() to register buffers.
#         # nn.Parameters require gradients by default.
#         self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(output_features))
#         else:
#             # You should always register all possible parameters, but the
#             # optional ones can be None if you want.
#             self.register_parameter('bias', None)

#         # Not a very smart way to initialize weights
#         self.weight.data.uniform_(-0.1, 0.1)
#         if bias is not None:
#             self.bias.data.uniform_(-0.1, 0.1)

#     def forward(self, input):
#         # See the autograd section for explanation of what happens here.
#         return LinearFunction.apply(input, self.weight, self.bias)

#     def extra_repr(self):
#         # (Optional)Set the extra information about this module. You can test
#         # it by printing an object of this class.
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )
