import os
import jax
from jax.experimental import jax2tf
import torch
import tensorflow as tf

from . import export


def exported_program_to_tf_function(ep: torch.export.ExportedProgram, enable_xla=False):
  jax_program = export.exported_program_to_jax_program(ep)

  example_inputs = jax_program.flatten_inputs(*jax_program.example_inputs)
  input_signature = [
      tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=f"args_{i}")
      for i, t in enumerate(example_inputs)
  ]
  tf_f = tf.function(
      jax2tf.convert(
          jax_program.flatten_callable, with_gradient=False, enable_xla=enable_xla
      ),
      autograph=False,
      input_signature=input_signature,
  )
  return tf_f


def exported_program_to_tf_module(
    ep: torch.export.ExportedProgram,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    function_alias: str = "",
    enable_xla=False,
):
  tfm = tf.Module()
  tfm.f = exported_program_to_tf_function(ep, enable_xla)

  return tfm


def save_exported_program_as_tf_saved_model(
    ep: torch.export.ExportedProgram,
    saved_model_dir: os.PathLike,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    function_alias: str = "",
    enable_xla=False,
):
  tfm = exported_program_to_tf_module(
      ep,
      serving_key=serving_key,
      function_alias=function_alias,
      enable_xla=enable_xla,
  )
  signatures = {serving_key: tfm.f.get_concrete_function(*tfm.f.input_signature)}
  save_options = tf.saved_model.SaveOptions(
      function_aliases={
          function_alias: tfm.f,
      }
  )
  tf.saved_model.save(
      tfm,
      saved_model_dir,
      signatures=signatures,
      options=save_options,
  )


def exported_program_to_tflite_flatbuffer(ep: torch.export.ExportedProgram):
  tfm = exported_program_to_tf_module(ep)
  tf_concrete_func = tfm.f.get_concrete_function(*tfm.f.input_signature)
  converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_concrete_func], tfm)
  tflite_model = converter.convert()
  return tflite_model
