import sys

sys.path.append(".")

import jax
import torch
import torch_jax as tj


@tj.hlfb.torch_impl("prod_add(Tensor a, Tensor b) -> Tensor", "prod_add", {})
def prod_add(a, b):
  return a @ b + b


class SampleModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.w = torch.ones((2, 2)) * 0.3

  def forward(self, a, b):
    x = prod_add(a, b)
    y = x + self.w
    return y


def main():
  m = SampleModel().eval()
  a, b = torch.ones((2, 2)), torch.ones((2, 2))
  args = (a, b)
  ep = torch.export.export(m, args)

  jax_f = tj.export.exported_program_to_jax_program(ep)
  jax_args = tj.map_jax(args)

  lowered = jax.jit(jax_f).lower(jax_args)

  print(lowered.as_text())
  # module @jit__unnamed_wrapped_function_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  #   func.func public @main(%arg0: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2xf32> {mhlo.layout_mode = "default"}) {
  #     %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x2xf32>
  #     %1 = call @prod_add.impl(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  #     %2 = stablehlo.custom_call @stablehlo.composite(%arg0, %arg1) {called_computations = [@prod_add.impl], composite.backend_config = {name = "prod_add"}} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  #     %3 = stablehlo.add %2, %0 : tensor<2x2xf32>
  #     return %3 : tensor<2x2xf32>
  #   }
  #   func.func private @prod_add.impl(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  #     %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  #     %1 = stablehlo.add %0, %arg1 : tensor<2x2xf32>
  #     return %1 : tensor<2x2xf32>
  #   }
  # }


if __name__ == "__main__":
  main()
