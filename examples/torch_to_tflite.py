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

  flatbuffer = tjtf.exported_program_to_tflite_flatbuffer(ep)
  interpreter = tf.lite.Interpreter(model_content=flatbuffer)
  interpreter.allocate_tensors()

  signature = list(interpreter.get_signature_list().values())[0]
  tflite_fn = interpreter.get_signature_runner()

  for _ in range(10):
    xs = [torch.rand(2, 2), torch.rand(2, 2)]
    torch_out = m(*xs).detach().numpy()

    np_xs = map(lambda x: x.detach().numpy(), xs)
    named_xs = {name: x for name, x in zip(signature["inputs"], xs)}
    tflite_out = tflite_fn(**named_xs)[signature["outputs"][0]]

    print("RMSE:", np.sqrt(np.mean((torch_out - tflite_out) ** 2)))


if __name__ == "__main__":
  main()
