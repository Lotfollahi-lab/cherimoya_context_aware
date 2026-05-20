Changelog
=========

v0.1.0
------

Model
~~~~~

* Added a fully fused **forward-only inference megakernel** for the
  Cheri Block: conv + norm + MLP + residual in two GPU passes, with
  bf16 dot products. Used automatically when
  ``torch.is_grad_enabled()`` is ``False`` and the MLP hidden width is
  a multiple of 16, with automatic fallback to the training Triton
  path otherwise. Numerically equivalent to the training path within
  ~1e-5 max-abs at unit-scale outputs, and roughly 1.9× faster than
  the training-fwd path on H200 at the default model size.
* The inference megakernel's bf16 weight cast is now materialized at
  ``.eval()`` time as non-persistent buffers and refreshed by a
  ``load_state_dict`` post-hook, instead of cached inside the
  compiled forward. This fixes a
  ``RuntimeError: accessing tensor output of CUDAGraphs that has been
  overwritten`` that previously surfaced when running multiple model
  instances or reloading weights mid-process, and removes the need
  for ``compile=False`` / ``compile_mode='max-autotune-no-cudagraphs'``
  as a workaround for that specific error. **User-visible
  consequence:** call ``model.eval()`` before inference to hit the
  fast path; the megakernel still runs without ``.eval()`` but
  recomputes the cast inline per call (adds ~10-27% at small batch,
  under ~2% at production batch). See :doc:`benchmarks` for the
  breakdown.
* Generalized the Kendall-Gal loss-weight parameters ``lw0`` and
  ``lw1`` from scalars to per-track vectors. ``lw0`` is now shape
  ``(n_outputs,)`` (one weight per profile track) and ``lw1`` is shape
  ``(n_count_outputs,)`` (one weight per count-head output). For
  single-task models both shapes are ``(1,)``, matching the format of
  every pre-vector checkpoint — existing single-task checkpoints load
  without changes. The freeze threshold now uses
  ``|grad(lw0)|.mean() < 1`` so it doesn't scale with track count.
  ``_mixture_loss`` correspondingly returns per-track loss vectors
  instead of scalars.
* The training Triton kernel and the CPU fallback are unchanged.
  Existing trained checkpoints are bit-compatible.
* Replaced the learnable channel-wise scaling with a fixed
  ``residual_scale`` constant (default 0.15).
* Added an exponential moving average (EMA) of model weights during
  training; validation and saved checkpoints use the EMA-applied
  weights.
* Changed the final profile convolution to ``kernel_width=1``.
* Set the default model size to 96 filters.
* Tuned the Muon and AdamW learning rates and weight decay values
  for improved convergence (Muon ``lr=0.025, wd=0.01``; AdamW
  ``lr=0.004, wd=0.2``).
* Best-model selection now monitors the validation count Pearson
  correlation rather than the total validation loss.

API
~~~

* ``Cherimoya.save`` / ``Cherimoya.load`` checkpoints now use a
  config + state_dict payload that is robust to source-layout
  changes and loads with PyTorch's ``weights_only=True``. Older
  pickle-based checkpoints (``torch.save(model, ...)``) are not
  compatible and must be migrated or retrained.
* :class:`cherimoya.cherimoya.EMA` is now a public top-level symbol
  alongside :class:`cherimoya.Cherimoya` and
  :class:`cherimoya.CheriBlock`.

Training
~~~~~~~~

* Default ``max_jitter`` for fitting lowered from 500 to 50.

Packaging and tooling
~~~~~~~~~~~~~~~~~~~~~

* Migrated from ``setup.py`` to ``pyproject.toml`` with ``uv``
  support.
* Refactored the CLI from a monolithic script into the
  ``cherimoya_cli`` modular package.
* Raised the minimum Python version to 3.10 and minimum PyTorch
  to 2.9.
* Added ``macs3``, ``bam2bw``, ``bpnet-lite``, ``triton``, and
  ``joblib`` as dependencies.
* Added a Sphinx documentation site hosted on Read the Docs.

v0.0.1
------

* Initial release of the Cherimoya model and pipeline.
* Includes the ``CheriBlock`` architecture and custom kernels.
* Features a dual-optimizer training strategy (AdamW + Muon).
* Implements a full end-to-end processing and modeling pipeline.
