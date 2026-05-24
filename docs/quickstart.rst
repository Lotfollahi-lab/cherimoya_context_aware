Quickstart
==========

Two paste-and-run examples — one for each interface. For full
walkthroughs see :doc:`tutorials/cli_pipeline` and
:doc:`tutorials/python_api`. New to the terms? Skim :doc:`glossary`.


Command-line pipeline
---------------------

For stranded ChIP-seq with input controls:

.. code-block:: bash

   cherimoya pipeline-json \
       -s hg38.fa -p peaks.narrowPeak \
       -i input.bam -c control.bam \
       -m JASPAR_2024.meme -n my_experiment -o pipeline.json

   cherimoya pipeline -p pipeline.json

This calls peaks with MACS3, converts BAMs to bigWigs, samples
GC-matched negatives, trains a Cherimoya model, computes attributions
via saturation mutagenesis, calls seqlets, annotates them with
tomtom-lite, and runs TF-MoDISco. All outputs land in the working
directory; the full output list and per-step descriptions are in
:doc:`tutorials/cli_pipeline`. Assay-specific recipes:
:doc:`recipes/chipseq_tf`, :doc:`recipes/atacseq`,
:doc:`recipes/dnaseq`.


Python API
----------

.. code-block:: python

   import torch
   from cherimoya import Cherimoya

   model = Cherimoya(n_filters=96, n_layers=9,
       signal_groups=[1]).cuda()
   X = torch.randn(2, 4, 2114, device="cuda")
   with torch.no_grad():
       y_profile, y_counts = model(X)

   print(y_profile.shape)   # torch.Size([2, 1, 1000])
   print(y_counts.shape)    # torch.Size([2, 1])

For a stranded ``(+, -)`` BPNet-style head — one group of two channels
that share a count prediction — pass ``signal_groups=[2]``. To co-train
an unstranded ATAC head alongside a stranded TF head, pass
``signal_groups=[1, 2]``: the profile head emits three channels (1 +
2) and the count head emits one prediction per group (1 for ATAC, 1
shared across the TF strands).

To one-hot encode real DNA, use ``tangermeme.utils.one_hot_encode``
(for a Python string) or ``tangermeme.io.extract_loci`` (for a FASTA
plus a BED of loci). To train from scratch with the same defaults the
CLI uses, see :doc:`tutorials/python_api`. To save and load
checkpoints, see :doc:`tutorials/save_load`. To compute attributions
or score variants, see :doc:`tutorials/attribution` and
:doc:`tutorials/variant_effect`.
