import sys

sys.path.append(".")

import torch
import torch_jax as tj


def main():
  a, b = tj.map_jax_tensor(torch.rand((2, 2)), torch.rand((2, 2)))
  print("--- a and b ---")
  print(a)
  print(b)
  print("--- a + b ---")
  x = a + b
  print(x)
  print("--- torch.nn.Softmax ---")
  print(torch.nn.Softmax(dim=-1)(x))


if __name__ == "__main__":
  main()
