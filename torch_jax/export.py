import copy
import torch
from torch.export import ExportedProgram
from torch.fx import _pytree as fx_pytree
from torch.utils import _pytree as pytree

from . import tensor
from . import lowering


class JaxFxInterpreter(torch.fx.Interpreter):

  def call_function(self, target, args, kwargs):
    return super().call_function(lowering.get_raw(target), args, kwargs)


def _extract_input_args(ep, args, kwargs):
  if (in_spec := ep.call_spec.in_spec) is not None:
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


def exported_program_to_jax_callable(ep):
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

  def run(args=None, kwargs=None):
    nonlocal ep, num_mutations
    args = args if args else tuple()
    kwargs = kwargs if kwargs else {}
    input_args = _extract_input_args(ep, args, kwargs)
    res = JaxFxInterpreter(ep.graph_module).run(
        *param_buffer_values,
        *input_args,
        *ordered_tensor_constants,
        enable_io_processing=False,
    )
    res = pytree.tree_unflatten(res, ep.call_spec.out_spec)
    return res

  return run
