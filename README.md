# Repository model update

Copy these files into the corresponding locations of the
`Knowledge-Distillation-and-Model-Compression-for-Fast-Denoising-of-CBCT-volumes`
repository.

```text
models/
  __init__.py
  HarUnet_model.py
  ResUnet_model.py
  README.md

scripts/
  distill.py
```

The supplied `scripts/distill.py` has been updated to import the models from
the `models` package.
