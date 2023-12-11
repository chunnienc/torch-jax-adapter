import uuid
import torch
import torch.library
import functools
from torch.utils._pytree import tree_map, tree_map_only
from typing import Callable

from . import tensor
from . import lowering

_torch_jax_custom_lib = torch.library.Library("torch_jax_custom", "DEF")


def define(spec: str, func: Callable, jax_lowering: Callable):
  parts = spec.split("(")
  name = parts[0] + "__" + uuid.uuid4().hex
  spec = "(".join([name, *parts[1:]])

  _torch_jax_custom_lib.define(spec)
  torch.library.impl(_torch_jax_custom_lib, name, "CompositeExplicitAutograd")(func)

  op = getattr(torch.ops.torch_jax_custom, name)
  lowering._register(op, func, jax_lowering)
  return op
