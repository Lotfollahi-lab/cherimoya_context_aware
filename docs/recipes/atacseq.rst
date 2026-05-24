Recipe: ATAC-seq
================

This recipe trains an unstranded Cherimoya model on paired-end
ATAC-seq, applying the standard +4 / −4 fragment shift to match the
Tn5 insertion offsets. ATAC-seq is typically modeled as a single
unstranded output track.


Inputs
------

* Reference genome FASTA (e.g. ``hg38.fa``).
* A BAM file of aligned ATAC-seq paired-end reads, **or** a BED/TSV
  fragment file.
* A motif database in MEME format.
* Optional: a BED of peak coordinates. If not provided, MACS3 will
  call peaks.


About the +4 / −4 shift
-----------------------

Tn5 transposes 9 bp apart and the standard correction many tools
apply is +4 / −5. Cherimoya's defaults and this recipe use **+4 / −4**
because the model treats the two end events symmetrically and the
−5 / −4 difference is not biologically meaningful at the resolution
Cherimoya predicts. Apply the shift here, not upstream, and **don't
double-shift** — if your BAM is already shifted, set the shifts to
zero.


Generate the pipeline JSON
--------------------------

For a paired-end BAM:

.. code-block:: bash

   cherimoya pipeline-json \
       -s hg38.fa \
       -i atac.bam \
       -m JASPAR_2024.meme \
       -n atac_experiment \
       -o atac.pipeline.json \
       -ps 4 -ns -4 -u -f -pe

Flag-by-flag:

* ``-ps 4`` / ``-ns -4`` — Tn5 shift on plus and minus ends.
* ``-u`` — unstranded output (single signal track).
* ``-f`` — treat the input as fragments. With ``-pe`` set, ``bam2bw``
  reads paired-end mates to reconstruct fragments before counting.
* ``-pe`` — paired-end. Causes MACS3 to use ``BAMPE`` file format
  rather than ``BAM``.

If your input is already a fragments TSV/BED (e.g. from
``snap-atac``, ``CellRanger``, or a custom ``samtools`` pipeline),
pass the fragment file with ``-i`` and drop ``-pe``; ``bam2bw``
detects the file extension and handles the fragment file format.


Run the pipeline
----------------

.. code-block:: bash

   cherimoya pipeline -p atac.pipeline.json

The steps mirror the ChIP-seq recipe, with these differences:

* MACS3 runs without a control file and with format ``BAMPE`` (or
  ``FRAG`` for fragment-file input).
* ``bam2bw`` is invoked with the ``-u`` (unstranded), ``-f``
  (fragments), and ``-ps 4 -ns -4`` flags, producing a single
  ``atac_experiment.bw`` rather than ``+.bw`` / ``-.bw`` pair.
* The trained Cherimoya model has ``signal_groups=[1]`` (one
  unstranded group) and ``n_control_tracks=0``.
* Attribution, seqlet calling, TF-MoDISco, and marginalization run
  the same way they do for ChIP-seq.


ATAC-seq peak counts can be noisier
-----------------------------------

ATAC-seq peaks span a wider range of summit heights than TF ChIP-seq.
Two parameters are worth checking after a first training run:

* ``fit_parameters.max_counts`` — if the training log shows very
  large gradient norms early in training, cap outlier peaks by
  setting this. The default ``PeakGenerator`` filter already drops
  peaks above ``1.2 × the 99th percentile`` of summed counts; setting
  ``max_counts`` adds an explicit ceiling on top of that.
* ``fit_parameters.max_jitter`` — the default of 50 is conservative.
  ATAC-seq peaks are typically wide enough that 128 is also safe and
  improves training-set diversity. Match this with ``in_window``
  large enough to absorb the jitter (``in_window`` is 2114 by
  default; ``max_jitter`` ≤ 128 fits comfortably).


Outputs
-------

See the "Outputs" table in :doc:`../tutorials/cli_pipeline`. ATAC-seq
runs produce one unstranded bigWig (``atac_experiment.bw``) rather
than a stranded pair, and the model has a single profile track. The
HTML reports under ``atac_experiment_modisco/`` and
``atac_experiment_marginalize/`` are the same format as ChIP-seq.
