# splits.py

"""Resolve a locus file into train/validation/test partitions.

``fit.py`` used to hand ``training_chroms``/``validation_chroms`` straight
to ``PeakGenerator``/``extract_loci`` and let ``tangermeme`` filter by
chromosome internally (``tangermeme/io.py:135-136``). That's fine for a
chromosome-holdout split, but it can't express a fold-based split (the
same chromosome spans train/val/test) or a split that's already been
decided upstream and baked into the locus file as a column.

``make_split_masks`` resolves all three cases the same way: load the loci
once, compute the train/val/test membership here, and return three
pre-filtered locus DataFrames. Callers pass those down to
``PeakGenerator``/``extract_loci`` with ``chroms=None`` -- the chromosome
concept is resolved above tangermeme, not inside it, for every mode
including ``"chrom"`` itself.

Three modes:

- ``"chrom"``: the existing behavior, re-expressed as a pre-filter. Loci
  are assigned by ``training_chroms``/``validation_chroms``/``test_chroms``
  membership, exactly the ``numpy.isin(df['chrom'], chroms)`` filter
  ``tangermeme`` applies internally (see ``filter_loci_by_chroms``, which
  is also what a caller should use to keep a negatives file's chromosome
  filtering consistent with this mode -- ``PeakGenerator`` has no separate
  negatives-chroms knob, so pre-filtering both peaks and negatives the
  same way is what preserves today's behavior once ``chroms=None`` is
  passed down). Training peaks keep every original column, not just
  chrom/start/end, so a narrowPeak summit column (``PeakGenerator(
  summits=True)``) survives the pre-filter intact.
- ``"fold"``: loci are assigned the fold of the ``fold_bed`` window
  (chrom/start/end/fold, e.g. scooby's ``sequences.bed``) containing their
  midpoint, then partitioned by ``val_folds``/``test_folds`` membership.
  Same rule ``context_aware_model/data/prep_bmmc.py``'s
  ``assign_folds_by_window`` uses to build ``peaks_with_split.bed``.
- ``"precomputed"``: the loci already carry a ``split_column`` (default
  ``"split"``) with values ``train``/``val``/``test`` -- partition on it
  directly.

Negatives are not fold- or split-aware in ``"fold"``/``"precomputed"``
mode: whoever prepares the negatives file is responsible for restricting
it to the training partition ahead of time (this is what
``context_aware_model``'s ``negatives_train_foldclean.bed`` already does).
Passing an unrestricted negatives file under ``"fold"``/``"precomputed"``
will silently include validation/test-fold negatives in training.
"""

import numpy
import pandas


_LOCI_COLS = [0, 1, 2]
_LOCI_NAMES = ['chrom', 'start', 'end']


def _load_loci(loci):
	"""Load ``loci`` into a single chrom/start/end DataFrame.

	Mirrors the column selection ``tangermeme.io._interleave_loci`` uses
	(the first three columns, positionally) so a chromosome filter
	computed here matches what extraction would otherwise have produced.
	``loci`` may be a path, a DataFrame, or a list of either; a list is
	concatenated -- row order does not need to match tangermeme's own
	interleaving, only the resulting row set does.
	"""

	if isinstance(loci, (str, pandas.DataFrame)):
		loci = [loci]

	dfs = []
	for df in loci:
		if isinstance(df, str):
			df = pandas.read_csv(df, sep='\t', usecols=_LOCI_COLS,
				header=None, index_col=False, names=_LOCI_NAMES)
		else:
			df = df.iloc[:, _LOCI_COLS].copy()
			df.columns = _LOCI_NAMES
		dfs.append(df)

	return pandas.concat(dfs, ignore_index=True)


def _load_raw(loci):
	"""Load ``loci`` preserving every column, chromosome always column 0
	positionally -- the same convention ``tangermeme`` itself uses for a
	DataFrame input. Unlike ``_load_loci`` this keeps columns past
	``end``, so a narrowPeak summit column (column 9, read when
	``PeakGenerator(summits=True)``) survives the pre-filter. Only the
	``"chrom"`` mode's training peaks need this -- negatives are always
	extracted with ``summits=False`` (``cherimoya/io.py:632``) and
	validation/test loci are never passed ``summits`` at all.
	"""

	if isinstance(loci, (str, pandas.DataFrame)):
		loci = [loci]

	dfs = [pandas.read_csv(df, sep='\t', header=None, index_col=False)
		if isinstance(df, str) else df.copy() for df in loci]
	df = pandas.concat(dfs, ignore_index=True)

	# Relabel the first three columns positionally, regardless of what
	# they were called coming in (a headerless read.csv leaves them
	# 0/1/2; a caller-supplied DataFrame may use anything) -- matches
	# _load_loci's output shape for chrom/start/end while leaving any
	# further columns (e.g. a summit column) exactly as they were.
	renamed = dict(zip(df.columns[:3], _LOCI_NAMES))
	return df.rename(columns=renamed)


def filter_loci_by_chroms(loci, chroms):
	"""Pre-filter ``loci`` to ``chroms``.

	The same ``numpy.isin(df['chrom'], chroms)`` mask
	``tangermeme.io.extract_loci`` applies internally, computed here so
	the filtered set can be passed down with ``chroms=None``.


	Parameters
	----------
	loci: str, pandas.DataFrame, or list of those
		The locus file(s) to filter.

	chroms: list
		The chromosomes to keep.


	Returns
	-------
	filtered: pandas.DataFrame
		A chrom/start/end DataFrame containing only loci on ``chroms``.
	"""

	df = _load_loci(loci)
	return df[numpy.isin(df['chrom'], chroms)].reset_index(drop=True)


def _split_chrom(loci, training_chroms, validation_chroms, test_chroms,
		fold_bed, val_folds, test_folds, split_column):
	raw = _load_raw(loci)
	train = raw[numpy.isin(raw['chrom'], training_chroms)].reset_index(drop=True)
	val = filter_loci_by_chroms(loci, validation_chroms)
	test = filter_loci_by_chroms(loci, test_chroms or [])
	return train, val, test


def _assign_folds(df, windows):
	"""Assign each locus the fold of the window containing its midpoint.

	``windows`` tiles the genome (contiguous, non-overlapping windows,
	one fold label per window) -- for each locus, find the window whose
	``[start, end)`` contains the locus midpoint, per chromosome, via a
	sorted binary search rather than an O(n_loci * n_windows) scan. Same
	rule as ``context_aware_model/data/prep_bmmc.py:assign_folds_by_window``.

	Loci whose chromosome has no windows, or whose midpoint falls in a
	gap between windows, get fold ``None``.
	"""

	fold = pandas.Series(index=df.index, dtype=object)
	midpoint = ((df['start'] + df['end']) // 2).to_numpy()

	for chrom, w in windows.groupby('chrom', sort=False):
		w = w.sort_values('start')
		starts = w['start'].to_numpy()
		ends = w['end'].to_numpy()
		folds = w['fold'].to_numpy()

		mask = (df['chrom'] == chrom).to_numpy()
		if not mask.any():
			continue

		mid = midpoint[mask]
		idx = numpy.searchsorted(starts, mid, side='right') - 1
		idx = numpy.clip(idx, 0, len(starts) - 1)
		contained = (mid >= starts[idx]) & (mid < ends[idx])

		fold.loc[df.index[mask]] = numpy.where(contained, folds[idx], None)

	return fold


def _split_fold(loci, training_chroms, validation_chroms, test_chroms,
		fold_bed, val_folds, test_folds, split_column):
	if not fold_bed:
		raise ValueError("split_mode='fold' requires 'fold_bed' (a "
			"chrom/start/end/fold BED, e.g. scooby's sequences.bed).")
	if not val_folds or not test_folds:
		raise ValueError("split_mode='fold' requires non-empty "
			"'val_folds' and 'test_folds'.")

	val_folds = [val_folds] if isinstance(val_folds, str) else list(val_folds)
	test_folds = [test_folds] if isinstance(test_folds, str) else list(test_folds)

	df = _load_loci(loci)
	windows = pandas.read_csv(fold_bed, sep='\t', header=None,
		names=['chrom', 'start', 'end', 'fold'])

	fold = _assign_folds(df, windows)
	df = df.loc[fold.notna()].copy()
	fold = fold.loc[fold.notna()]

	is_val = fold.isin(val_folds).to_numpy()
	is_test = fold.isin(test_folds).to_numpy()
	is_train = ~(is_val | is_test)

	train = df.loc[is_train].reset_index(drop=True)
	val = df.loc[is_val].reset_index(drop=True)
	test = df.loc[is_test].reset_index(drop=True)
	return train, val, test


def _split_precomputed(loci, training_chroms, validation_chroms, test_chroms,
		fold_bed, val_folds, test_folds, split_column):
	if isinstance(loci, str):
		full = pandas.read_csv(loci, sep='\t', header=None,
			names=_LOCI_NAMES + ['name', split_column])
	elif isinstance(loci, pandas.DataFrame):
		if split_column not in loci.columns:
			raise ValueError("split_mode='precomputed' requires a {!r} "
				"column in loci".format(split_column))
		full = loci
	else:
		raise ValueError("split_mode='precomputed' requires loci to be a "
			"path or DataFrame with a {!r} column, not a list"
			.format(split_column))

	train = full.loc[full[split_column] == 'train', _LOCI_NAMES].reset_index(drop=True)
	val = full.loc[full[split_column] == 'val', _LOCI_NAMES].reset_index(drop=True)
	test = full.loc[full[split_column] == 'test', _LOCI_NAMES].reset_index(drop=True)
	return train, val, test


SPLIT_MODES = {
	'chrom': _split_chrom,
	'fold': _split_fold,
	'precomputed': _split_precomputed,
}


def make_split_masks(loci, split_mode, training_chroms=None, validation_chroms=None,
		test_chroms=None, fold_bed=None, val_folds=None, test_folds=None,
		split_column='split'):
	"""Resolve ``loci`` into train/validation/test locus DataFrames.


	Parameters
	----------
	loci: str, pandas.DataFrame, or list of those
		The locus file(s) to split. ``"precomputed"`` mode requires a
		single path or DataFrame, not a list.

	split_mode: str
		One of ``"chrom"``, ``"fold"``, ``"precomputed"``. See the module
		docstring for what each mode does.

	training_chroms, validation_chroms, test_chroms: list or None, optional
		Chromosome membership for ``"chrom"`` mode. ``test_chroms`` may be
		empty or ``None`` -- the returned test set is then empty. Ignored
		by the other modes.

	fold_bed: str or None, optional
		Path to a chrom/start/end/fold BED (e.g. scooby's
		``sequences.bed``) tiling the genome with fold labels. Required by
		``"fold"`` mode; ignored otherwise.

	val_folds, test_folds: str, list, or None, optional
		Fold label(s) assigned to validation / test in ``"fold"`` mode.
		Required (non-empty) by ``"fold"`` mode; ignored otherwise.

	split_column: str, optional
		The column name carrying ``train``/``val``/``test`` labels in
		``"precomputed"`` mode. Default is ``"split"``.


	Returns
	-------
	train, val, test: pandas.DataFrame
		Three locus DataFrames, pre-filtered to their partition (chrom,
		start, end -- plus any further original columns for ``"chrom"``
		mode's ``train``, see above). Pass these down to
		``PeakGenerator``/``extract_loci`` with ``chroms=None``.
	"""

	if split_mode not in SPLIT_MODES:
		raise ValueError("Unknown split_mode {!r}; choose from {}".format(
			split_mode, sorted(SPLIT_MODES)))

	return SPLIT_MODES[split_mode](loci, training_chroms, validation_chroms,
		test_chroms, fold_bed, val_folds, test_folds, split_column)
