# PyTorch-JAX Adapter

## Action Items

### Initialization

- [x] Port Han's implementation of [jax integration](https://github.com/qihqi/gpt-fast/blob/jax_experiment/jax_integration.py)

### General

- [ ] Setup package requirements
- [ ] Setup pip package
- [ ] Port Aten op test suite from `torch_xla`

### Export

- [x] Add export API to StableHLO MLIR ([example](examples/torch_to_stablehlo.py))
- [x] Add export API to TF concrete function ([example](examples/torch_to_tf.py))
- [x] Add export API to TF SavedModel ([example](examples/torch_to_tf_saved_model.py))
- [x] Add export API to TFLite flatbuffer ([example](examples/torch_to_tflite.py))
- [ ] Quantization Q/DQ ops
- [ ] StableHLO Composite (HLFB)
- [ ] Custom lowering

### ODML

- [ ] Integrate with `odmlbench`
- [ ] Add example for running and exporting torchvision resnet18 (including all missing ops).


## Reference

- Han Qi, "PyTorch to Jax", PyTorch Dev Mini Summit, Dec 2023 [[Link]](https://docs.google.com/presentation/d/1LDlmhsNQzD5qljv25Xg_ej2ygZU2PglnAyx5IvDqS8Q/edit?resourcekey=0-ycJqMuiT6vf7i0hzhJpY7g#slide=id.g2a370ce899e_2_5)
