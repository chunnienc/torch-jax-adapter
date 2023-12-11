import sys

sys.path.append(".")

import torch
import numpy as np
import tensorflow as tf
import torch_jax as tj
import torch_jax.tf_integration as tjtf


class SampleModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.w = torch.ones((2, 2)) * 0.3
    self.layer = torch.nn.Softmax(dim=-1)

  def forward(self, a, b):
    x = a * 2 + b
    y = x + self.w
    z = self.layer(y.T)
    return z


def main():
  m = SampleModel().eval()
  a, b = torch.ones((2, 2)), torch.ones((2, 2))
  args = (a, b)
  ep = torch.export.export(m, args)

  tf_f = tjtf.exported_program_to_tf_function(ep)

  print(tf_f(np.random.rand(2, 2), np.random.rand(2, 2)))


if __name__ == "__main__":
  main()
