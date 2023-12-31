import functools
import torch
import torch.nn.functional as F
from torch._decomp import core_aten_decompositions
from torch.utils._pytree import tree_map, tree_map_only
import jax
from jax import numpy as jnp
import dataclasses
from typing import Callable


from . import tensor

_lowerings = {}
_decompositions = core_aten_decompositions()


@dataclasses.dataclass
class LoweringRegistry:
  op: Callable
  call_torch: Callable
  call_jax: Callable


def _register(op, call_torch: Callable, call_jax: Callable):
  if isinstance(op, torch._ops.OpOverloadPacket):
    for overload in op.overloads():
      op_overload = getattr(op, overload)
      _lowerings[op_overload] = LoweringRegistry(op, call_torch, call_jax)
  else:
    _lowerings[op] = LoweringRegistry(call_torch, call_jax)


def _jax_io_to_jax_tensor_func(f, *args, **kwargs):
  args = tree_map(tensor.strict_to_jax_tensor, args)
  kwargs = tree_map(tensor.strict_to_jax_tensor, kwargs)
  res = f(*args, **kwargs)
  if isinstance(res, (tuple, list, dict, set)):
    return tree_map_only(tensor.to_jax, lambda t: t._elem, res)
  return tensor.to_jax(res)


def _jax_tensor_io_to_jax_func(f, *args, **kwargs):
  try:
    args = jax.tree_util.tree_map(tensor.strict_to_jax, args)
    kwargs = jax.tree_util.tree_map(tensor.strict_to_jax, kwargs)
    res = f(*args, **kwargs)
    if isinstance(res, (tuple, list)):
      return tree_map_only(jnp.ndarray, tensor.JaxTensor, res)
    return tensor.JaxTensor(res)
  except Exception as e:
    raise RuntimeError(f"Failed to execute JAX lowering {f}") from e


def register_torch(op):
  def inner(func):
    _register(
        op,
        call_torch=functools.partial(_jax_tensor_io_to_jax_func, func),
        call_jax=func,
    )
    return func

  return inner


def get(op):
  if op in _lowerings:
    return _lowerings[op]
  # if op in _decompositions:
  #   func = _decompositions[op]
  #   return LoweringRegistry(
  #       torch_op=func,
  #       jax_func=functools.partial(_jax_tensor_io_to_jax_func, func),
  #   )
  raise ValueError(f"Lowering not found: {op}")


def _get_numpy_dtype(dtype):
  return {
      torch.double: jnp.double,
      torch.float32: jnp.float32,
      # torch.half: jnp.half,
      torch.long: jnp.int64,
      torch.int32: jnp.int32,
      torch.int16: jnp.int16,
      torch.bool: jnp.bool_,
  }.get(dtype)


@register_torch(torch.ops.aten.view)
@register_torch(torch.ops.aten._unsafe_view)
def _aten_unsafe_view(x, shape):
  return jnp.reshape(x, shape)


@register_torch(torch.ops.aten.add)
def _aten_add(x, y):
  """
  if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
      assert x.dtype == y.dtype, (x.dtype, y.dtype)
  """
  return x + y


@register_torch(torch.ops.aten.copy_)
def _aten_copy(x, y, memory_format=None):
  return jnp.copy(y)


@register_torch(torch.ops.aten.clone)
def _aten_clone(x, memory_format=None):
  return jnp.copy(x)


@register_torch(torch.ops.aten.full)
def _aten_full(size, value, **kwargs):
  return jnp.full(size, value)


@register_torch(torch.ops.aten.index_copy)
def _aten_index_copy(x, dim, indexes, source):
  # return jax.lax.scatter(x, index, dim)
  dims = []
  for i in range(len(x.shape)):
    if i == dim:
      dims.append(indexes)
    else:
      dims.append(slice(None, None, None))
  return x.at[dim].set(source)


@register_torch(torch.ops.aten.select)
def _aten_select(x, dim, index):
  """
  slice_sizes = list(x.shape)
  slice_sizes[dim] = 1
  indexes = jnp.append(indexes, 1)
  offset_dims = [i for i in range(len(x.shape)) if i != dim]
  gather_dnums = jax.lax.GatherDimensionNumbers(
      offset_dims=(dim, ),
      collapsed_slice_dims=(dim, ),
      start_index_map=(dim, ),
  )
  return jax.lax.gather(x, indexes, gather_dnums, tuple(slice_sizes))
  """
  dims = []
  for i in range(len(x.shape)):
    if i == dim:
      dims.append(index)
    else:
      dims.append(slice(None, None, None))
  return x[tuple(dims)]


@register_torch(torch.ops.aten.index_select)
def _aten_index_select(x, dim, indexes):
  """
  slice_sizes = list(x.shape)
  slice_sizes[dim] = 1
  indexes = jnp.append(indexes, 1)
  offset_dims = [i for i in range(len(x.shape)) if i != dim]
  gather_dnums = jax.lax.GatherDimensionNumbers(
      offset_dims=(dim, ),
      collapsed_slice_dims=(dim, ),
      start_index_map=(dim, ),
  )
  return jax.lax.gather(x, indexes, gather_dnums, tuple(slice_sizes))
  """
  dims = []
  for i in range(len(x.shape)):
    if i == dim:
      dims.append(indexes)
    else:
      dims.append(slice(None, None, None))
  return x[tuple(dims)]


@register_torch(torch.ops.aten.mean)
def _aten_mean(x, dim, keepdim):
  return jnp.mean(x, dim, keepdims=keepdim)


@register_torch(torch.ops.aten.sub)
def _aten_sub(x, y):
  return x - y


@register_torch(torch.ops.aten.mm)
def _aten_mm(x, y):
  res = x @ y
  return res


@register_torch(torch.ops.aten.mul)
def _aten_mul(x, y):
  return x * y


@register_torch(torch.ops.aten.silu)
def _aten_silu(x):
  return jax.nn.silu(x)


@register_torch(torch.ops.aten.t)
def _aten_t(x):
  return jnp.transpose(x)


@register_torch(torch.ops.aten.transpose)
def _aten_transpose(x, dim0, dim1):
  shape = list(range(len(x.shape)))
  shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
  return jnp.transpose(x, shape)


@register_torch(torch.ops.aten.triu)
def _aten_triu(m, k):
  return jnp.triu(m, k)


@register_torch(torch.ops.aten.slice)
def _aten_slice(self, dim=0, start=None, end=None, step=1):
  sl = slice(start, end, step)
  dims = []
  for i in range(len(self.shape)):
    if i == dim:
      dims.append(sl)
    else:
      dims.append(slice(None, None, None))
  return self[tuple(dims)]


@register_torch(torch.ops.aten.detach)
def _aten_detach(self):
  return self


@register_torch(torch.ops.aten.view_as_real)
def _aten_view_as_real(x):
  real = jnp.real(x)
  im = jnp.imag(x)
  res = jnp.stack([real, im], -1)
  return res


@register_torch(torch.ops.aten.stack)
def _aten_stack(tensors, dim=0):
  return jnp.stack(tensors, dim)


@register_torch(torch.ops.aten._softmax)
def _aten_softmax(x, dim, halftofloat):
  return jax.nn.softmax(x, dim)


@register_torch(torch.ops.aten.pow)
def _aten_pow(x, y):
  if isinstance(y, int):
    y = float(y)
  if isinstance(y, jnp.ndarray):
    y = y.astype(jnp.astype(jnp.bfloat16))
  return jnp.power(x, y)


@register_torch(torch.ops.aten.view_as_complex)
def _aten_view_as_complex(input):
  if input.dtype == jnp.bfloat16:
    input = input.astype(jnp.float32)
  x, y = input[..., 0], input[..., 1]
  return jax.lax.complex(x, y)


@register_torch(torch.ops.aten.div)
def _aten_div(x, y):
  return x / y


@register_torch(torch.ops.aten.bmm)
def _aten_bmm(x, y):
  res = x @ y
  assert res.dtype == jnp.bfloat16
  return res
  # return jnp.einsum('bnm,bmk->bnk', x, y)


@register_torch(torch.ops.aten.embedding)
def _aten_embedding(a, w):
  return jnp.take(a, w, axis=0)


@register_torch(torch.ops.aten.rsqrt)
def _aten_rsqrt(x):
  return jax.lax.rsqrt(x)


@register_torch(torch.ops.aten.expand)
def _aten_expand(x, dims):
  return jnp.broadcast_to(x, dims)


@register_torch(torch.ops.aten.dot)
def _aten_dot(x, y):
  return jnp.dot(x, y)


@register_torch(torch.ops.aten._to_copy)
def _aten__to_copy(tensor, **kwargs):
  dtype = _get_numpy_dtype(kwargs["dtype"])
  if dtype != tensor.dtype:
    return tensor.astype(dtype)
  return jnp.copy(tensor)


@register_torch(torch.ops.aten.empty)
def _aten_empty(sizes, **kwargs):
  return jnp.zeros(sizes)


@register_torch(torch.ops.aten.index_put_)
def _aten_index_put(self, indexes, values):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  return self.at[indexes].set(values)


@register_torch(torch.ops.aten.index)
def _aten_index(self, indexes):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  return self[indexes]


import jax.numpy as jnp
import jax.lax as lax


@register_torch(torch.ops.aten.split_with_sizes)
def split_with_sizes(x, sizes, dim):
  """Splits an array `x` into sub-arrays based on static sizes `sizes`.

  Args:
    x: The input array to split.
    sizes: A 1D array of integer sizes for each sub-array.

  Returns:
    A list of sub-arrays.
  """
  rank = x.ndim
  splits = np.cumsum(sizes)  # Cumulative sum for split points

  def make_range(rank, dim, start, end):
    res = [slice(None, None, None)] * rank
    res[dim] = slice(start, end)
    return tuple(res)

  return [
      x[make_range(rank, dim, start, end)]
      for start, end in zip([0] + list(splits[:-1]), splits)
  ]


@register_torch(torch.ops.aten.permute)
def permute(t, dims):
  return jnp.transpose(t, dims)


@register_torch(torch.ops.aten.unsqueeze)
def _aten_unsqueeze(self, dim):
  if dim < 0:
    dim += input.ndim
  return jnp.expand_dims(self, dim)
