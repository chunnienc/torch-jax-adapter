import sys

sys.path.append(".")

import functools
import jax
import torch
from jax.interpreters import xla
from jax._src import core as jax_core
from jax._src.lib.mlir.dialects import hlo
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
import torch_jax as tj

# Create _rms_norm_fwd_p for forward operation.
run_magic_p = jax_core.Primitive("run_magic")
run_magic_p.def_impl(functools.partial(xla.apply_primitive, run_magic_p))


def run_magic_lowering(ctx, x):
  custom_call = hlo.CustomCallOp(
      [x.type],
      [x],
      call_target_name=ir.StringAttr.get("run_magic"),
  )
  return custom_call.results


mlir.register_lowering(run_magic_p, run_magic_lowering)


def run_magic_abstract(x):
  return jax_core.ShapedArray(x.shape, x.dtype, named_shape=x.named_shape)


run_magic_p.def_abstract_eval(run_magic_abstract)


def run_magic_jax(x):
  return run_magic_p.bind(x)


def run_magic_meta(x):
  return torch.empty_like(x)


@tj.custom_op.define("run_magic(Tensor x) -> Tensor", run_magic_jax, run_magic_meta)
def run_magic(x):
  import math

  xx = x.clone()
  assert len(x.shape) == 2
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      xx[i, j] = -100 if math.exp(xx[i, j]) < 0.2 else 100

  return xx


class SampleModel(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, a, b):
    return a * 2 + run_magic(b)


def main():
  m = SampleModel().eval()
  a, b = torch.ones((2, 2)), torch.ones((2, 2))
  args = (a, b)
  print("Run with torch and Jax backend:", m(*args))
  ep = torch.export.export(m, args)

  jax_f = tj.export.exported_program_to_jax_program(ep)
  jax_args = tj.map_jax(args)

  lowered = jax.jit(jax_f).lower(jax_args)

  print(lowered.as_text())
  # module @jit__unnamed_wrapped_function_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  #   func.func public @main(%arg0: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2xf32> {mhlo.layout_mode = "default"}) {
  #     %0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  #     %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  #     %2 = stablehlo.multiply %arg0, %1 : tensor<2x2xf32>
  #     %3 = stablehlo.custom_call @run_magic(%arg1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  #     %4 = stablehlo.add %2, %3 : tensor<2x2xf32>
  #     return %4 : tensor<2x2xf32>
  #   }
  # }


if __name__ == "__main__":
  main()
