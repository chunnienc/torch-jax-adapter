import uuid
import torch
import torch.library
import functools
from torch.utils._pytree import tree_map, tree_map_only
from typing import Callable

from . import tensor
from . import lowering

_torch_jax_custom_lib = torch.library.Library("torch_jax_custom", "DEF")


def _define(spec: str, func: Callable, call_jax: Callable, meta_func=None):
  parts = spec.split("(")
  name = parts[0] + "__" + uuid.uuid4().hex
  spec = "(".join([name, *parts[1:]])

  _torch_jax_custom_lib.define(spec)
  torch.library.impl(_torch_jax_custom_lib, name, "CompositeExplicitAutograd")(func)
  if meta_func is not None:
    torch.library.impl(_torch_jax_custom_lib, name, "Meta")(meta_func)

  op = getattr(torch.ops.torch_jax_custom, name)
  lowering._register(op, func, call_jax)
  return op


def define(spec, call_jax: Callable, meta_func=None):
  return functools.partial(_define, spec, call_jax=call_jax, meta_func=meta_func)
