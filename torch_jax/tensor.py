import torch
from torch.utils._pytree import tree_map, tree_map_only
import jax
from jax import numpy as jnp
from . import lowering

_jax_device = jax.devices()[0]


def to_jax(t):
  import torch.utils.dlpack as torchdl
  from jax import dlpack as jaxdl

  if isinstance(t, JaxTensor):
    return t._elem
  if isinstance(t, torch.Tensor):
    t = t.detach()
    if hasattr(t, "_elem"):
      return t._elem

    if t.dtype == torch.bool:
      t = t.to(torch.int32)
    dl = torchdl.to_dlpack(t)
    jt = jaxdl.from_dlpack(dl)
    return jax.device_put(jt, device=_jax_device)
  return t


def strict_to_jax(t):
  """Same as to_jax but rejects regular torch tensor."""
  if isinstance(t, torch.Tensor) and not hasattr(t, "_elem"):
    raise ValueError("torch.Tensor is not accepted.")
  return to_jax(t)


def to_jax_tensor(t):
  t = to_jax(t)
  if isinstance(t, jnp.ndarray):
    return JaxTensor(t)
  return t


def strict_to_jax_tensor(t):
  """Same as to_jax but rejects regular torch tensor."""
  if isinstance(t, torch.Tensor) and not hasattr(t, "_elem"):
    raise ValueError("torch.Tensor is not accepted.")
  return to_jax_tensor(t)


def map_jax(*ts):
  ts = tree_map(to_jax, ts)
  if len(ts) == 1:
    return ts[0]
  return ts


def map_jax_tensor(*ts):
  ts = tree_map(to_jax_tensor, ts)
  if len(ts) == 1:
    return ts[0]
  return ts


# All of the tensor examples in this zoo inherit from BaseTensor. Ideally,
# however, they would inherit directly from Tensor. This is just our staging
# ground for applying behavior that hasn't yet made it into core but that
# we would like to apply by default.
class JaxTensor(torch.Tensor):
  # See https://github.com/pytorch/pytorch/pull/73727 ; this is necessary
  # to ensure that super().__new__ can cooperate with each other
  @staticmethod
  def __new__(cls, elem):
    return torch.Tensor._make_subclass(
        cls,
        torch.empty(elem.shape, dtype=torch.float32, device="meta"),
        require_grad=False,
    )

  def __init__(self, elem):
    super().__init__()
    self._elem = elem

  def __str__(self):
    return f"[JaxTensor({type(self._elem)})] {self._elem}"

  def __jax_array__(self):
    return self._elem

  @property
  def shape(self):
    return self._elem.shape

  @property
  def ndim(self):
    return len(self._elem.shape)

  def flatten(self, start_dim=0, end_dim=-1):
    if end_dim == -1:
      end_dim = self.ndim
    new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim:]
    return torch.reshape(self, new_shape)

  def __setitem__(self, key, val):
    if isinstance(key, tuple):
      key = tuple(to_jax(k) if isinstance(k, torch.Tensor) else k for k in key)
    else:
      key = key._elem
    self._elem = self._elem.at[key].set(val._elem)

  def type_as(self, other):
    self._elem = self._elem.astype(other._elem.dtype)
    return self

  __torch_function__ = torch._C._disabled_torch_function_impl

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    # print('running...', func.name())
    res = lowering.get(func).call_torch(*args, **kwargs)
    # run_torch_and_diff(func, args, kwargs, res)
    if func.name() == "aten::copy_":
      args[0]._elem = res._elem
      return args[0]
    return res
