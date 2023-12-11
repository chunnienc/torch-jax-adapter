import sys

sys.path.append(".")

import jax
import torch
import torch_jax as tj


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

  jax_f = tj.export.exported_program_to_jax_program(ep)
  jax_args = tj.map_jax(args)

  lowered = jax.jit(jax_f).lower(jax_args)

  print(lowered.as_text())
  # module @jit_run attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  #   func.func public @main(%arg0: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
  #     %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x2xf32>
  #     %1 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  #     %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  #     %3 = stablehlo.multiply %arg0, %2 : tensor<2x2xf32>
  #     %4 = stablehlo.add %3, %arg1 : tensor<2x2xf32>
  #     %5 = stablehlo.add %4, %0 : tensor<2x2xf32>
  #     %6 = stablehlo.transpose %5, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  #     %7 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  #     %8 = stablehlo.reduce(%6 init: %7) applies stablehlo.maximum across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  #     %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
  #     %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
  #     %11 = stablehlo.subtract %6, %10 : tensor<2x2xf32>
  #     %12 = stablehlo.exponential %11 : tensor<2x2xf32>
  #     %13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  #     %14 = stablehlo.reduce(%12 init: %13) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
  #     %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
  #     %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x2xf32>
  #     %17 = stablehlo.divide %12, %16 : tensor<2x2xf32>
  #     return %17 : tensor<2x2xf32>
  #   }
  # }


if __name__ == "__main__":
  main()
