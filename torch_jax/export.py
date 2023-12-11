import copy
import dataclasses
import torch
from torch.export import ExportedProgram
from torch.fx import _pytree as fx_pytree
from torch.utils import _pytree as pytree
from jax import numpy as jnp
from typing import Sequence, Tuple

from . import tensor
from . import lowering


class JaxFxInterpreter(torch.fx.Interpreter):

  def call_function(self, target, args, kwargs):
    return super().call_function(lowering.get(target).call_jax, args, kwargs)


@dataclasses.dataclass
class JaxProgram:
  exported_program: torch.export.ExportedProgram
  param_buffer_values: Tuple[jnp.ndarray]
  ordered_tensor_constants: Tuple[jnp.ndarray]

  def __hash__(self):
    return hash(self.exported_program)

  @property
  def example_inputs(self):
    args, kwargs = self.exported_program.example_inputs
    args = pytree.tree_map(tensor.to_jax, args)
    kwargs = pytree.tree_map(tensor.to_jax, kwargs)
    return args, kwargs

  def flatten_inputs(self, args, kwargs):
    if args is None:
      args = tuple()
    if kwargs is None:
      kwargs = {}

    if (in_spec := self.exported_program.call_spec.in_spec) is not None:
      if (
          in_spec.type == tuple
          and len(in_spec.children_specs) == 2
          and in_spec.children_specs[0].type == tuple
          and in_spec.children_specs[1].type == dict
      ):
        # NOTE: this is the case where in_spec is for both args and kwargs
        return fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
      return fx_pytree.tree_flatten_spec(args, in_spec)
    return copy.deepcopy(args)

  def unflatten_outputs(self, res):
    return pytree.tree_unflatten(res, self.exported_program.call_spec.out_spec)

  def __call__(self, args=None, kwargs=None):
    inputs = self.flatten_inputs(args, kwargs)
    outputs = JaxFxInterpreter(self.exported_program.graph_module).run(
        *self.param_buffer_values,
        *inputs,
        *self.ordered_tensor_constants,
        enable_io_processing=False,
    )
    return self.unflatten_outputs(outputs)

  @property
  def flatten_callable(self):
    def func(*inputs: jnp.ndarray):
      nonlocal self
      return JaxFxInterpreter(self.exported_program.graph_module).run(
          *self.param_buffer_values,
          *inputs,
          *self.ordered_tensor_constants,
          enable_io_processing=False,
      )

    return func


def exported_program_to_jax_program(ep) -> JaxProgram:
  ep = ep.run_decompositions()

  param_buffer_keys = ep.graph_signature.parameters + ep.graph_signature.buffers
  param_buffer_values = tuple(ep.state_dict[key] for key in param_buffer_keys)

  if hasattr(ep.graph_signature, "lifted_tensor_constants"):
    ordered_tensor_constants = tuple(
        ep.tensor_constants[name] for name in ep.graph_signature.lifted_tensor_constants
    )
  else:
    ordered_tensor_constants = tuple()

  num_mutations = len(ep.graph_signature.buffers_to_mutate)

  param_buffer_values = pytree.tree_map(tensor.to_jax, param_buffer_values)
  ordered_tensor_constants = pytree.tree_map(tensor.to_jax, ordered_tensor_constants)

  return JaxProgram(ep, param_buffer_values, ordered_tensor_constants)
