Benchmarks
==========

This page reports measured timing and numerical agreement for the three
forward paths of the Cheri Block. See :doc:`architecture` for the
description of each path.

Setup
-----

All numbers below were measured on a single NVIDIA H200 with PyTorch
2.9 and Triton 3.5. CPU numbers were measured on the same host's CPU
cores. Each measurement uses a warmup phase to lock in Triton autotune
configurations before the timed iterations begin; reported numbers are
the median over many iterations.

The benchmark script is checked into the repository root as
``bench_kernels.py``. It is excluded from the package install. Run it
on your own hardware to get numbers specific to your machine.

Forward timing
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Path
     - Single block
       (N=16, L=1024, C=96)
     - Full Cherimoya
       (N=4, L=2114, default 9-layer 96-filter)
   * - CPU (pure PyTorch)
     - 4.71 ms
     - 21.92 ms
   * - GPU, grad-enabled (training fwd)
     - 0.101 ms
     - 1.106 ms
   * - GPU, ``no_grad`` (inference megakernel)
     - **0.064 ms**
     - **0.583 ms**

The megakernel is dispatched automatically when ``torch.is_grad_enabled()``
is ``False`` and the MLP hidden width (``expansion * n_filters``) is a
multiple of 16. Training is unaffected.

All three paths agree on the model output to ~1e-5 max-abs at
unit-scale outputs, so trained checkpoints produce numerically
equivalent predictions through every path. The breakdown of which
pair agrees to fp32 precision vs. which pair differs by the bf16
weight cast is in :doc:`architecture`.


Caveats
-------

* The numbers above are for *forward* passes only. Training step time
  is dominated by the backward pass and the optimizer step, neither of
  which the megakernel touches.
* H200-specific. On older GPUs (A100, V100) the absolute numbers
  change, but the relative ordering CPU ≫ train-fwd > inference-fwd
  holds.
* CPU inference is comfortable for development and one-off evaluation
  on a laptop; only training and high-throughput inference benefit
  from a GPU.
* The autotune step on the first call to any Triton kernel takes
  significant wall time. The warmup pass in ``bench_kernels.py``
  exists specifically to amortize this. If you call the model once
  and observe a slow first call, that is autotune — subsequent calls
  use the cached configuration.
