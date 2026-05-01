CLI Pipeline Tutorial
=====================

This tutorial walks through using the Cherimoya command-line tool to run the
full end-to-end pipeline from raw sequencing data to motif analysis.


Prerequisites
-------------

Before starting, make sure you have:

- Cherimoya installed (see :doc:`../installation`)
- A reference genome FASTA file (e.g., ``hg38.fa``)
- Signal files (BAM, SAM, BED, or bigWig format)
- (Optional) Control signal files


Overview of CLI Commands
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``negatives``
     - Sample GC-content-matched negative regions
   * - ``pipeline-json``
     - Generate a pipeline configuration JSON file
   * - ``fit``
     - Train a Cherimoya model
   * - ``evaluate``
     - Evaluate a trained model
   * - ``attribute``
     - Calculate sequence attributions
   * - ``seqlets``
     - Identify important subsequences from attributions
   * - ``marginalize``
     - Run marginalization experiments for motifs
   * - ``pipeline``
     - Run the full end-to-end pipeline
   * - ``batch``
     - Run multiple pipelines in parallel


Step 1: Generate a Pipeline Configuration
------------------------------------------

The pipeline is driven by a JSON configuration file. You can generate one using
the ``pipeline-json`` command:

.. code-block:: bash

   cherimoya pipeline-json \
       -s /path/to/hg38.fa \
       -i /path/to/signal.bam \
       -n my_ctcf_experiment \
       -o ctcf.pipeline.json

For stranded experiments with controls:

.. code-block:: bash

   cherimoya pipeline-json \
       -s /path/to/hg38.fa \
       -i /path/to/treatment.bam \
       -c /path/to/control.bam \
       -n ctcf_stranded \
       -o ctcf_stranded.pipeline.json

.. tip::

   Use ``-u`` for unstranded data and ``-f`` if your input consists of
   fragment files rather than aligned reads.

.. tip::

   If using ATAC-seq or DNase-seq data you may need to shift the reads.
   Many packages use +4/-5 for ATAC-seq shifting, but the recommended shift
   here is actually +4/-4. You can use -ps and -ns to shift your data but
   make sure the original data is not shifted!


Step 2: Customize the Parameters (Optional)
--------------------------------------------

Open the generated JSON file and adjust parameters as needed. Key parameters
include:

.. code-block:: json

   {
       "fit_parameters": {
           "n_filters": 64,
           "n_layers": 9,
           "max_epochs": 50,
           "lr": 0.004,
           "batch_size": 512,
           "max_jitter": 500,
           "early_stopping": 15
       }
   }

.. note::

   The pipeline will automatically call peaks with MACS3, convert BAMs to
   bigWigs, and generate GC-matched negatives if these are not explicitly
   provided in the JSON.


Step 3: Run the Pipeline
------------------------

.. code-block:: bash

   cherimoya pipeline -p ctcf.pipeline.json

The pipeline will execute the following steps in order:

1. **Peak calling** (if ``loci`` is not provided) — uses MACS3
2. **Data conversion** (if signal files are BAMs) — uses bam2bw
3. **Negative sampling** (if ``negatives`` is not provided) — GC-matched
4. **Model training** — trains a Cherimoya model with dual optimizers
5. **Attribution calculation** — saturation mutagenesis on validation chroms
6. **Seqlet identification** — extract important subsequences
7. **TF-MoDISco** — motif discovery and report generation
8. **Marginalization** — motif effect size estimation


Running Individual Steps
------------------------

You can also run steps individually using their respective JSON configuration
files.

**Training:**

.. code-block:: bash

   cherimoya fit -p fit_parameters.json

**Evaluation:**

.. code-block:: bash

   cherimoya evaluate -p evaluate_parameters.json

**Attribution:**

.. code-block:: bash

   cherimoya attribute -p attribute_parameters.json


Batch Processing
----------------

To train models on many datasets in parallel, use the ``batch`` command:

.. code-block:: bash

   cherimoya batch -p batch_parameters.json

The batch command will automatically distribute jobs across available GPUs.
Use ``"device": "*"`` in the JSON to auto-detect all available CUDA devices.

.. code-block:: json

   {
       "name": null,
       "device": "*",
       "signals": "/path/to/data/*.bam",
       "sequences": "/path/to/hg38.fa"
   }

When ``signals`` contains a glob pattern and ``name`` is null, names are
automatically derived from the signal filenames.
