import jax
from jax import linear_util as lu
from jax._src import core as jax_core
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir import ir
from jax.interpreters import mlir
from jax import tree_util
import dataclasses
from jax import api_util
from typing import Dict, Any, Callable
from jax import numpy as jnp
import functools

from . import custom_op
from . import lowering


def _composite_impl(f, *args, **kwargs):
  with jax_core.new_sublevel():
    return f.call_wrapped(*args)


composite_p = jax_core.CallPrimitive("composite")
composite_p.def_impl(_composite_impl)


# The call_composite function can be used to wrap an abstraction as a composite
# which has the semantics of a call, but shows up as a custom call with a
# FuncOp implementation in the exported StableHLO:
#   def mySquared(x):
#     return x * x
#   def myMain():
#     return call_composite(mySquared, 2, name="my.squared")
def _call_composite(f, *args, name: str, attributes: dict[str, Any] = {}, **kwargs):
  fun = lu.wrap_init(f, kwargs)
  flat_args, in_tree = tree_util.tree_flatten(args)
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
  out_flat = composite_p.bind(flat_fun, *flat_args, name=name, attributes=attributes)
  return tree_util.tree_unflatten(out_tree(), out_flat)


# Make custom_call which calls the implementation function
# Currently this leaks a CallOp since we're using the `core_call_lowering`
# function, but this should get cleaned up by DCE easily.
def _composite_stablehlo_lowering(ctx, *args, name, call_jaxpr, attributes, **kwargs):
  impl = mlir.core_call_lowering(ctx, *args, name=name + ".impl", call_jaxpr=call_jaxpr)
  call_op = impl[0][0].owner
  called_fn = call_op.attributes["callee"]

  custom_call = hlo.CustomCallOp(
      [r.type for r in call_op.results],
      call_op.operands,
      call_target_name=ir.StringAttr.get("stablehlo.composite"),
      called_computations=ir.ArrayAttr.get([called_fn]),
  )
  composite_attrs = {"name": ir.StringAttr.get(name)}
  if len(attributes) > 0:
    composite_attrs["attributes"] = ir.DictAttr.get(attributes)
  custom_call.attributes["composite.backend_config"] = ir.DictAttr.get(composite_attrs)
  return custom_call.results


mlir.register_lowering(composite_p, _composite_stablehlo_lowering)


# TODO: Use xla_mark_pattern like custom_op to implement composite builder
# TODO: Support scalar attributes
def torch_impl(spec: str, name: str, attributes: Dict[str, Any]):
  def register(torch_func):
    jax_func = functools.partial(lowering._jax_io_to_jax_tensor_func, torch_func)

    def jax_lowering(*args, **kwargs):
      if kwargs:
        raise ValueError("StableHLO Composite impl does not accept keyword arguments.")
      for arg in args:
        if not isinstance(arg, jnp.ndarray):
          raise ValueError("StableHLO Composite impl does not accept non-Jax tensors.")
      # return _call_composite(torch_func, *args, name=name, attributes=attributes)
      return _call_composite(jax_func, *args, name=name, attributes=attributes)

    return custom_op.define(spec, torch_func, jax_lowering)

  return register
