"""Tests for cherimoya.splits: the train/validation/test split resolver
used by ``cherimoya fit``."""

import pandas
import pytest

from cherimoya.splits import filter_loci_by_chroms, make_split_masks


_ROWS = [
	("chr1", 100, 200),
	("chr2", 100, 200),
	("chr2", 300, 400),
	("chr8", 100, 200),
	("chr9", 300, 400),
	("chr20", 500, 600),
	("chrX", 10, 20),
]


def _write_bed(path, rows):
	path.write_text("\n".join("{}\t{}\t{}".format(*r) for r in rows) + "\n")
	return str(path)


def _sorted(df):
	return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


# --- "chrom" mode: backward compatibility -----------------------------

def test_chrom_mode_matches_tangermeme_filter_str_path(tmp_path):
	"""split_mode='chrom' must select the exact same rows tangermeme's own
	internal filter would -- this is the guarantee every existing
	Cherimoya config relies on, since chrom mode is the default and the
	filter now runs as a pre-extraction mask instead of inside
	extract_loci."""

	from tangermeme.io import _interleave_loci

	bed = _write_bed(tmp_path / "loci.bed", _ROWS)
	training_chroms = ["chr2", "chr9", "chrX"]
	validation_chroms = ["chr8", "chr20"]

	old_train = _interleave_loci(bed, chroms=training_chroms)[["chrom", "start", "end"]]
	old_val = _interleave_loci(bed, chroms=validation_chroms)[["chrom", "start", "end"]]

	new_train, new_val, new_test = make_split_masks(bed, "chrom",
		training_chroms=training_chroms, validation_chroms=validation_chroms)

	pandas.testing.assert_frame_equal(
		_sorted(old_train), _sorted(new_train[["chrom", "start", "end"]]))
	pandas.testing.assert_frame_equal(
		_sorted(old_val), _sorted(new_val))
	assert len(new_test) == 0


def test_chrom_mode_matches_tangermeme_filter_dataframe(tmp_path):
	"""Same equivalence check, but with an in-memory DataFrame ``loci``
	instead of a path -- PeakGenerator/extract_loci accept both."""

	from tangermeme.io import _interleave_loci

	df = pandas.DataFrame(_ROWS, columns=["chrom", "start", "end"])
	chroms = ["chr2", "chr9", "chrX"]

	old = _interleave_loci(df, chroms=chroms)[["chrom", "start", "end"]]
	new_train, new_val, new_test = make_split_masks(df, "chrom",
		training_chroms=chroms, validation_chroms=[])

	pandas.testing.assert_frame_equal(
		_sorted(old), _sorted(new_train[["chrom", "start", "end"]]))


def test_chrom_mode_preserves_summit_column(tmp_path):
	"""Training peaks must keep every original column so a narrowPeak
	summit column (read by extract_loci when PeakGenerator(summits=True))
	survives the pre-filter -- only chrom/start/end are needed for the
	filter itself."""

	rows = [("chr2", 100, 200, "peak1", 0, ".", 0, 0, 0, 42),
		("chr8", 100, 200, "peak2", 0, ".", 0, 0, 0, 7)]
	bed = tmp_path / "narrowpeak.bed"
	bed.write_text("\n".join("\t".join(str(c) for c in r) for r in rows) + "\n")

	train, val, test = make_split_masks(str(bed), "chrom",
		training_chroms=["chr2"], validation_chroms=["chr8"])

	assert train.shape[1] == 10
	assert train.iloc[0, 9] == 42


def test_chrom_mode_empty_test_chroms_is_backward_compatible(tmp_path):
	"""An absent/empty test_chroms (today's only configuration) must
	leave train/val untouched and produce an empty test set."""

	bed = _write_bed(tmp_path / "loci.bed", _ROWS)

	train, val, test = make_split_masks(bed, "chrom",
		training_chroms=["chr2", "chr9", "chrX"],
		validation_chroms=["chr8", "chr20"],
		test_chroms=None)

	assert len(test) == 0
	assert len(train) == 4   # chr2 (x2), chr9, chrX
	assert len(val) == 2     # chr8, chr20


def test_chrom_mode_negatives_filter_matches_peaks_filter(tmp_path):
	"""filter_loci_by_chroms (used in fit.py to pre-filter negatives for
	'chrom' mode) must select the same rows PeakGenerator's shared
	`chroms` kwarg would have today."""

	bed = _write_bed(tmp_path / "negatives.bed", _ROWS)
	filtered = filter_loci_by_chroms(bed, ["chr2", "chr9"])
	assert sorted(filtered["chrom"]) == ["chr2", "chr2", "chr9"]


# --- "fold" mode --------------------------------------------------------

def test_fold_mode_assigns_by_midpoint_in_window():
	# Windows: chr1 tiled by three 100bp folds; chr2 has a gap.
	windows = pandas.DataFrame([
		("chr1", 0, 100, "foldA"),
		("chr1", 100, 200, "foldB"),
		("chr1", 200, 300, "foldC"),
		("chr2", 0, 50, "foldA"),
		# chr2 [50, 100) is a gap -- no window covers it.
	], columns=["chrom", "start", "end", "fold"])

	peaks = pandas.DataFrame([
		("chr1", 10, 30),    # midpoint 20 -> foldA
		("chr1", 140, 160),  # midpoint 150 -> foldB
		("chr1", 240, 260),  # midpoint 250 -> foldC
		("chr2", 10, 30),    # midpoint 20 -> foldA
		("chr2", 60, 80),    # midpoint 70 -> no window, dropped
	], columns=["chrom", "start", "end"])

	from cherimoya.splits import _assign_folds

	fold = _assign_folds(peaks, windows)
	assert list(fold) == ["foldA", "foldB", "foldC", "foldA", None]


def test_fold_mode_partitions_train_val_test(tmp_path):
	(tmp_path / "sequences.bed").write_text(
		"chr1\t0\t100\tfoldA\n"
		"chr1\t100\t200\tfoldB\n"
		"chr1\t200\t300\tfoldC\n"
		"chr1\t300\t400\tfoldD\n"
	)
	loci_bed = _write_bed(tmp_path / "loci.bed", [
		("chr1", 10, 30), ("chr1", 140, 160),
		("chr1", 240, 260), ("chr1", 340, 360),
	])

	train, val, test = make_split_masks(loci_bed, "fold",
		fold_bed=str(tmp_path / "sequences.bed"),
		val_folds=["foldB"], test_folds=["foldC"])

	assert len(train) == 2   # foldA, foldD
	assert len(val) == 1     # foldB
	assert len(test) == 1    # foldC

	# The test partition must be disjoint from train and val -- the
	# "held out of training" guarantee.
	all_rows = pandas.concat([train, val, test])
	assert len(all_rows) == len(all_rows.drop_duplicates())


def test_fold_mode_requires_fold_bed(tmp_path):
	loci_bed = _write_bed(tmp_path / "loci.bed", _ROWS)
	with pytest.raises(ValueError, match="fold_bed"):
		make_split_masks(loci_bed, "fold", val_folds=["a"], test_folds=["b"])


def test_fold_mode_requires_val_and_test_folds(tmp_path):
	loci_bed = _write_bed(tmp_path / "loci.bed", _ROWS)
	(tmp_path / "sequences.bed").write_text("chr1\t0\t100\tfoldA\n")
	with pytest.raises(ValueError, match="val_folds"):
		make_split_masks(loci_bed, "fold",
			fold_bed=str(tmp_path / "sequences.bed"))


# --- "precomputed" mode --------------------------------------------------

def test_precomputed_mode_partitions_on_split_column(tmp_path):
	rows = [
		("chr1", 0, 100, "peak1", "train"),
		("chr1", 100, 200, "peak2", "val"),
		("chr1", 200, 300, "peak3", "test"),
		("chr2", 0, 100, "peak4", "train"),
	]
	path = tmp_path / "peaks_with_split.bed"
	path.write_text("\n".join("\t".join(str(c) for c in r) for r in rows) + "\n")

	train, val, test = make_split_masks(str(path), "precomputed")

	assert len(train) == 2
	assert len(val) == 1
	assert len(test) == 1
	assert list(test.iloc[0]) == ["chr1", 200, 300]

	# Held out of training: the test row's coordinates must not appear
	# in train.
	assert not ((train["chrom"] == "chr1") & (train["start"] == 200)).any()


def test_precomputed_mode_accepts_dataframe_and_custom_split_column():
	df = pandas.DataFrame([
		("chr1", 0, 100, "fold3"),
		("chr1", 100, 200, "fold4"),
		("chr1", 200, 300, "fold0"),
	], columns=["chrom", "start", "end", "partition"])

	train, val, test = make_split_masks(df, "precomputed", split_column="partition")

	# None of "fold3"/"fold4"/"fold0" match "train"/"val"/"test" literally
	# under the default label convention, so this also exercises that
	# make_split_masks doesn't coerce or guess -- only exact matches count.
	assert len(train) + len(val) + len(test) == 0


def test_precomputed_mode_requires_split_column():
	df = pandas.DataFrame([("chr1", 0, 100)], columns=["chrom", "start", "end"])
	with pytest.raises(ValueError, match="split"):
		make_split_masks(df, "precomputed")


def test_precomputed_mode_rejects_list_of_loci(tmp_path):
	a = _write_bed(tmp_path / "a.bed", _ROWS)
	b = _write_bed(tmp_path / "b.bed", _ROWS)
	with pytest.raises(ValueError):
		make_split_masks([a, b], "precomputed")


# --- dispatch -------------------------------------------------------------

def test_unknown_split_mode_raises(tmp_path):
	bed = _write_bed(tmp_path / "loci.bed", _ROWS)
	with pytest.raises(ValueError, match="split_mode"):
		make_split_masks(bed, "nonsense")
