import sys

sys.path.append(".")

import os
import tempfile
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

  with tempfile.TemporaryDirectory() as tmpdirname:
    saved_model = os.path.join(tmpdirname, "m")
    tjtf.save_exported_program_as_tf_saved_model(ep, saved_model)

    tf_m = tf.saved_model.load(saved_model)

  for _ in range(10):
    xs = [torch.rand(2, 2), torch.rand(2, 2)]
    torch_out = m(*xs).detach().numpy()

    np_xs = map(lambda x: x.detach().numpy(), xs)
    tf_out = tf_m.f(*np_xs)[0]
    print("RMSE:", np.sqrt(np.mean((torch_out - tf_out) ** 2)))


if __name__ == "__main__":
  main()
