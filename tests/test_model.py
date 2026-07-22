"""Tests for the Cherimoya model construction and forward pass."""

import os

import pytest
import torch

from cherimoya import Cherimoya


@pytest.fixture
def small_model_kwargs():
	# A tiny but valid configuration that runs quickly on CPU.
	return dict(n_filters=8, n_layers=3, signal_groups=[1], n_control_tracks=0,
		verbose=False)


def _input_window_for(model):
	"""Compute a valid input window length for `model`."""
	# Output window must be > 0; trimming bytes are removed from each side.
	return 2 * model.trimming + 64


# --------- Construction ----------------------------------------------------

def test_default_construction():
	model = Cherimoya(verbose=False)
	assert model.n_filters == 128
	assert model.n_layers == 9
	assert model.n_outputs == 1
	assert model.n_control_tracks == 0
	assert model.expansion == 2
	assert model.residual_scale == 0.15
	# Default trimming is 46 + sum_{i<n_layers} 2**i = 46 + 511 = 557.
	assert model.trimming == 46 + sum(2**i for i in range(9))


def test_residual_scale_propagates_to_blocks():
	model = Cherimoya(n_filters=8, n_layers=3, residual_scale=0.07,
		verbose=False)
	assert model.residual_scale == 0.07
	for block in model.blocks:
		assert block.residual_scale == 0.07


def test_residual_scale_round_trips_through_save_load(tmp_path):
	model = Cherimoya(n_filters=8, n_layers=2, residual_scale=0.42,
		verbose=False)
	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.residual_scale == 0.42
	for block in loaded.blocks:
		assert block.residual_scale == 0.42


def test_expansion_propagates_to_blocks():
	model = Cherimoya(n_filters=8, n_layers=2, expansion=3, verbose=False)
	for block in model.blocks:
		assert block.expansion == 3
		assert block.linear1.out_features == 24
		assert block.linear2.in_features == 24


def test_expansion_round_trips_through_save_load(tmp_path):
	model = Cherimoya(n_filters=8, n_layers=2, expansion=4, verbose=False)
	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.expansion == 4
	for block in loaded.blocks:
		assert block.linear1.out_features == 32


def test_custom_construction(small_model_kwargs):
	model = Cherimoya(**small_model_kwargs)
	assert model.n_filters == 8
	assert model.n_layers == 3
	assert len(model.blocks) == 3


def test_profile_head_and_control_tracks_shape():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 1],
		n_control_tracks=2, verbose=False)
	assert model.fconv.out_channels == 2
	assert model.fconv.in_channels == 8 + 2  # n_filters + n_control_tracks


# --------- signal_groups construction --------------------------------------

def test_signal_groups_default_is_single_unstranded():
	model = Cherimoya(n_filters=8, n_layers=2, verbose=False)
	assert model.signal_groups == [1]
	assert model.n_outputs == 1
	assert model.n_groups == 1
	assert model.lw0.shape == (1,)
	assert model.lw1.shape == (1,)


def test_signal_groups_grouped_pair():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False)
	assert model.signal_groups == [1, 2]
	assert model.n_outputs == 3   # 1 + 2 channels total
	assert model.n_groups == 2    # 2 groups
	assert model.fconv.out_channels == 3
	# Count head is per-group; lw0 and lw1 are *both* per-group so each
	# modality contributes equally to the Kendall-Gal loss.
	assert model.linear.out_features == 2
	assert model.lw0.shape == (2,)
	assert model.lw1.shape == (2,)


def test_signal_groups_stranded_pair_shares_one_count_prediction():
	"""A stranded ``(+, -)`` pair is one group, so the count head emits
	a single per-group prediction tying the two strands together."""

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[2],
		verbose=False)
	assert model.n_outputs == 2   # two profile channels
	assert model.n_groups == 1    # one group
	assert model.linear.out_features == 1
	assert model.lw1.shape == (1,)


def test_signal_groups_rejects_bad_values():
	with pytest.raises(ValueError, match="positive ints"):
		Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 0],
			verbose=False)
	with pytest.raises(ValueError, match="positive ints"):
		Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, -1],
			verbose=False)


def test_signal_groups_rejects_empty_list():
	"""An empty signal_groups would construct a model with zero output
	channels — degenerate. The constructor must catch this at the
	boundary rather than letting Conv1d / Linear error 20 lines later
	with a less helpful message."""

	with pytest.raises(ValueError, match="non-empty"):
		Cherimoya(n_filters=8, n_layers=2, signal_groups=[], verbose=False)


def test_grouped_model_forward_loss_measures_integrate():
	"""End-to-end shape contract for a co-trained grouped model: the
	model's forward, the mixture loss, and the performance measures
	must all line up on the same ``signal_groups=[1, 2]`` config. This
	is what would catch a mismatch where, say, the count head was
	sized off n_outputs and the loss off n_groups."""

	from cherimoya.losses import _mixture_loss
	from cherimoya.performance import calculate_performance_measures

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False, compile=False).eval()
	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	X = torch.randn(4, 4, L)
	y = torch.randint(0, 5, (4, 3, out_L)).float()

	with torch.no_grad():
		y_hat_logits, y_hat_logcounts = model(X)

	# Profile head emits sum(signal_groups) channels; count head emits
	# len(signal_groups) predictions.
	assert y_hat_logits.shape == (4, 3, out_L)
	assert y_hat_logcounts.shape == (4, 2)

	profile_loss, count_loss = _mixture_loss(
		y, y_hat_logits, y_hat_logcounts, signal_groups=[1, 2])
	# Per-group on both sides now: one loss term per modality.
	assert profile_loss.shape == (2,)
	assert count_loss.shape == (2,)
	assert torch.isfinite(profile_loss).all()
	assert torch.isfinite(count_loss).all()

	measures = calculate_performance_measures(
		y_hat_logits, y, y_hat_logcounts,
		measures=['count_pearson', 'count_mse'],
		signal_groups=[1, 2])
	# One Pearson correlation per group.
	assert measures['count_pearson'].shape == (2,)
	assert torch.isfinite(measures['count_pearson']).all()


def test_grouped_model_fit_smoke(tmp_path):
	"""Run ``Cherimoya.fit`` for a few iterations on a grouped model
	(``signal_groups=[1, 2]``) with synthetic data. Confirms the
	training loop, validation pass, optimizer routing, save callback,
	and ``_mixture_loss`` / ``calculate_performance_measures``
	plumbing all agree on shapes end-to-end. CPU-only and ~seconds."""

	import torch
	import os
	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR
	from cherimoya.io import (PeakNegativeSampler,
		channel_permutation_from_groups)

	signal_groups = [1, 2]
	n_outputs = sum(signal_groups)

	model = Cherimoya(n_filters=8, n_layers=2,
		signal_groups=signal_groups, verbose=False, compile=False)
	# .save() writes next to model.name; redirect into tmp_path so the
	# test doesn't pollute the working directory.
	model.name = str(tmp_path / "smoke")

	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	# Synthetic peaks + negatives. Use deterministic seeds so the test
	# can't flake from random initialization quirks.
	g = torch.Generator().manual_seed(0)
	n_peaks, n_negs = 16, 8
	peak_sequences = torch.zeros(n_peaks, 4, L)
	peak_sequences[:, 0, :] = 1.0  # one-hot at A
	peak_signals = torch.randint(0, 5, (n_peaks, n_outputs, out_L),
		generator=g).float()
	neg_sequences = torch.zeros(n_negs, 4, L)
	neg_sequences[:, 0, :] = 1.0
	neg_signals = torch.zeros(n_negs, n_outputs, out_L)

	sampler = PeakNegativeSampler(
		peak_sequences=peak_sequences, peak_signals=peak_signals,
		negative_sequences=neg_sequences, negative_signals=neg_signals,
		in_window=L, out_window=out_L, max_jitter=0,
		negative_ratio=0, random_state=0, reverse_complement=True,
		signal_perm=channel_permutation_from_groups(signal_groups))
	training_data = torch.utils.data.DataLoader(sampler, batch_size=4,
		num_workers=0)

	# Optimizers — split params the same way fit.py does: 2D projection
	# weights to Muon, Kendall lw0/lw1 to SGD, everything else (incl.
	# conv_weight and the head) to AdamW.
	muon_params, adam_params, lw_params = [], [], []
	for name, p in model.named_parameters():
		if name in ("lw0", "lw1"):
			lw_params.append(p)
		elif (p.ndim == 2 and "weight" in name and name != "linear.weight"
				and "conv_weight" not in name):
			muon_params.append(p)
		else:
			adam_params.append(p)
	muon_opt = Muon(muon_params, lr=1e-3, weight_decay=0.0)
	adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)
	lw_opt = torch.optim.SGD(lw_params, lr=1e-3, weight_decay=0.0,
		momentum=0.9)
	muon_sched = LinearLR(muon_opt, start_factor=1.0, total_iters=1)
	adam_sched = LinearLR(adam_opt, start_factor=1.0, total_iters=1)
	lw_sched = LinearLR(lw_opt, start_factor=1.0, total_iters=1)

	# Validation tensors with the same shape contract.
	X_valid = torch.zeros(4, 4, L)
	X_valid[:, 0, :] = 1.0
	y_valid = torch.randint(0, 5, (4, n_outputs, out_L),
		generator=g).float()

	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		best = model.fit(training_data, muon_opt, adam_opt, lw_opt,
			muon_sched, adam_sched, lw_sched,
			X_valid=X_valid, X_ctl_valid=None, y_valid=y_valid,
			max_epochs=2, batch_size=4, dtype='float32',
			device='cpu', early_stopping=None)
	finally:
		os.chdir(cwd)

	# `best` is the validation count-Pearson at the best epoch (a
	# numpy/torch scalar after `nan_to_num`). It just needs to come
	# back as a finite real number — NaN would indicate a degenerate
	# count target shape.
	import math
	assert math.isfinite(float(best))

	# Both log files exist. The summary log has the original column
	# set; the detail log appends one ProfilePearson/CountPearson
	# column per signal group (here [1, 2] -> two groups -> four
	# extra columns).
	summary_log = tmp_path / "smoke.log"
	detail_log = tmp_path / "smoke.detailed.log"
	assert summary_log.exists(), "summary log not written"
	assert detail_log.exists(), "detail log not written"

	summary_header = summary_log.read_text().splitlines()[0].split("\t")
	detail_header = detail_log.read_text().splitlines()[0].split("\t")

	# Detail header strictly extends the summary header.
	assert detail_header[:len(summary_header)] == summary_header
	extra = detail_header[len(summary_header):]
	assert extra == [
		"ProfilePearson_g0", "ProfilePearson_g1",
		"CountPearson_g0", "CountPearson_g1",
	], "unexpected detail columns: {}".format(extra)


def test_signal_groups_round_trips_through_save_load(tmp_path):
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False)
	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.signal_groups == [1, 2]
	assert loaded.n_outputs == 3
	assert loaded.n_groups == 2


@pytest.mark.parametrize("signal_groups,expected_lw0,expected_lw1", [
	([1],       (1,), (1,)),  # single unstranded (the default)
	([1, 1, 1], (3,), (3,)),  # three unstranded groups — one weight per group
	([1, 2],    (2,), (2,)),  # mixed: one profile loss term per *group*
	([2],       (1,), (1,)),  # one stranded pair: 2 profile channels share one loss
])
def test_lw0_lw1_shapes(signal_groups, expected_lw0, expected_lw1):
	"""Both `lw0` and `lw1` size with `len(signal_groups)` — every signal
	group contributes one profile-loss term and one count-loss term
	regardless of channel count, so the uncertainty weights are
	per-group on both sides of the Kendall-Gal combination."""

	model = Cherimoya(n_filters=8, n_layers=2,
		signal_groups=signal_groups, verbose=False)
	assert model.lw0.shape == expected_lw0
	assert model.lw1.shape == expected_lw1
	# Both initialize to ones; the fit loop relies on this.
	assert torch.allclose(model.lw0, torch.ones(*expected_lw0))
	assert torch.allclose(model.lw1, torch.ones(*expected_lw1))


@pytest.mark.parametrize("signal_groups", [
	[1],          # single unstranded (the default)
	[1, 1, 1],    # all unstranded
	[1, 2],       # mixed
])
def test_lw0_lw1_save_load_round_trip(tmp_path, signal_groups):
	"""Save/load preserves lw0/lw1 shapes and values across all
	grouping shapes."""

	model = Cherimoya(n_filters=8, n_layers=2,
		signal_groups=signal_groups, verbose=False)
	# Perturb the weights so the round-trip comparison is non-trivial.
	with torch.no_grad():
		model.lw0.add_(torch.randn_like(model.lw0) * 0.1)
		model.lw1.add_(torch.randn_like(model.lw1) * 0.1)

	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)

	assert loaded.lw0.shape == model.lw0.shape
	assert loaded.lw1.shape == model.lw1.shape
	assert torch.allclose(loaded.lw0, model.lw0)
	assert torch.allclose(loaded.lw1, model.lw1)


def test_default_name_includes_filters_and_layers():
	model = Cherimoya(n_filters=12, n_layers=4, verbose=False)
	assert model.name == "cherimoya.12.4"


# --------- Forward pass ----------------------------------------------------

def test_forward_shape_no_controls(small_model_kwargs):
	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(2, 4, L)
	y_profile, y_counts = model(X)
	assert y_profile.shape == (2, 1, L - 2 * model.trimming)
	assert y_counts.shape == (2, 1)


def test_forward_with_controls():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		n_control_tracks=2, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	X_ctl = torch.randn(1, 2, L)
	y_profile, y_counts = model(X, X_ctl)
	assert y_profile.shape == (1, 1, L - 2 * model.trimming)
	assert y_counts.shape == (1, 1)


def test_forward_multi_output_per_track_counts():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 1, 1],
		n_control_tracks=0, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	y_profile, y_counts = model(X)
	assert y_profile.shape == (1, 3, L - 2 * model.trimming)
	assert y_counts.shape == (1, 3)


def test_forward_runs_on_default_device(device, small_model_kwargs):
	model = Cherimoya(**small_model_kwargs).to(device).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L, device=device)
	y_profile, y_counts = model(X)
	assert y_profile.device.type == device
	assert y_counts.device.type == device


# --------- Save / load round-trip -----------------------------------------

def test_save_load_roundtrip_preserves_predictions(tmp_path, small_model_kwargs):
	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	expected_profile, expected_counts = model(X)

	path = tmp_path / "model.torch"
	model.save(str(path))

	loaded = Cherimoya.load(str(path), compile=False).eval()
	assert loaded.n_filters == model.n_filters
	assert loaded.n_layers == model.n_layers
	got_profile, got_counts = loaded(X)
	assert torch.allclose(expected_profile, got_profile, atol=1e-6)
	assert torch.allclose(expected_counts, got_counts, atol=1e-6)


def test_save_payload_format(tmp_path, small_model_kwargs):
	model = Cherimoya(**small_model_kwargs)
	path = tmp_path / "model.torch"
	model.save(str(path))
	# Must be loadable in weights_only mode — the security-safe path.
	payload = torch.load(str(path), weights_only=True, map_location='cpu')
	assert isinstance(payload, dict)
	assert set(payload.keys()) == {'config', 'state_dict'}
	assert payload['config']['n_filters'] == small_model_kwargs['n_filters']


def test_load_to_specified_device(tmp_path, small_model_kwargs, device):
	model = Cherimoya(**small_model_kwargs)
	path = tmp_path / "model.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), device=device, compile=False)
	# Check at least one parameter ended up on the requested device.
	param = next(loaded.parameters())
	assert param.device.type == device


# --------- Inference fast-path invariants ---------------------------------
#
# These tests guard against silent regressions when a separate inference
# kernel is dispatched under `torch.no_grad()`. Existing trained-model
# checkpoints must continue to produce the same predictions, so the
# tolerance budget here is tight on fp32.

def test_model_no_grad_matches_grad_cpu(small_model_kwargs):
	"""On CPU the model takes the same code path regardless of grad
	state — verify that explicitly so the merge of an inference-only
	kernel doesn't accidentally divert CPU through a different
	branch."""

	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(2, 4, L)

	y_profile_grad, y_counts_grad = model(X)
	with torch.no_grad():
		y_profile_ng, y_counts_ng = model(X)

	assert torch.allclose(y_profile_grad.detach(), y_profile_ng,
		atol=1e-6, rtol=1e-5)
	assert torch.allclose(y_counts_grad.detach(), y_counts_ng,
		atol=1e-6, rtol=1e-5)


def test_model_save_load_predictions_match_under_no_grad(tmp_path,
	small_model_kwargs):
	"""The inference-time entry point is: load a saved model, set
	`.eval()`, run forward under `torch.no_grad()`. This is exactly the
	combination a separate inference kernel would dispatch under, so the
	round-trip must produce the same predictions as a grad-enabled
	forward on the unloaded model. Tight tolerance is required because
	users rely on saved checkpoints being numerically stable."""

	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)

	expected_profile, expected_counts = model(X)

	path = tmp_path / "model.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False).eval()

	with torch.no_grad():
		got_profile, got_counts = loaded(X)

	assert torch.allclose(expected_profile, got_profile, atol=1e-6)
	assert torch.allclose(expected_counts, got_counts, atol=1e-6)


def test_model_no_grad_matches_grad_with_controls():
	"""Same parity check, exercising the control-tracks branch of the
	forward where the inference path's residual layout could in
	principle differ."""

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		n_control_tracks=2, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	# Counts head takes log(sum(X_ctl)+1); use non-negative controls so
	# that the comparison values are finite.
	X_ctl = torch.rand(1, 2, L)

	y_profile_grad, y_counts_grad = model(X, X_ctl)
	with torch.no_grad():
		y_profile_ng, y_counts_ng = model(X, X_ctl)

	assert torch.allclose(y_profile_grad.detach(), y_profile_ng,
		atol=1e-6, rtol=1e-5)
	assert torch.allclose(y_counts_grad.detach(), y_counts_ng,
		atol=1e-6, rtol=1e-5)


def test_model_small_n_filters_works_under_no_grad():
	"""With n_filters=8 and expansion=1, the per-block hidden width is 8
	— below the multiple-of-16 constraint that a fused inference kernel
	may impose. Such configurations must transparently fall back."""

	model = Cherimoya(n_filters=8, n_layers=2, expansion=1, signal_groups=[1],
		n_control_tracks=0, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)

	with torch.no_grad():
		y_profile, y_counts = model(X)

	assert y_profile.shape == (1, 1, L - 2 * model.trimming)
	assert y_counts.shape == (1, 1)
	assert torch.isfinite(y_profile).all()
	assert torch.isfinite(y_counts).all()


@pytest.mark.cuda
@pytest.mark.triton
def test_model_no_grad_matches_grad_cuda():
	"""GPU parity at the model level. The inference path of an
	individual CheriBlock is correctness-checked in test_cheri.py; this
	test validates that stacking multiple blocks plus the head layers
	does not compound errors past the 1e-4 budget for fp32 inputs."""

	model = Cherimoya(n_filters=32, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda().eval()
	L = _input_window_for(model)
	X = torch.randn(2, 4, L, device='cuda')

	y_profile_grad, y_counts_grad = model(X)
	with torch.no_grad():
		y_profile_ng, y_counts_ng = model(X)

	prof_diff = (y_profile_grad - y_profile_ng).abs().max().item()
	count_diff = (y_counts_grad - y_counts_ng).abs().max().item()
	assert prof_diff <= 1e-4, f"profile diverged: max-abs-diff={prof_diff}"
	assert count_diff <= 1e-4, f"counts diverged: max-abs-diff={count_diff}"


@pytest.mark.cuda
@pytest.mark.triton
def test_model_save_load_no_grad_matches_grad_cuda(tmp_path):
	"""The full inference workflow on GPU: build, save, reload, eval,
	predict under no_grad. Must match the grad-enabled forward of the
	original (unloaded) model within tight tolerance — this is the
	contract trained checkpoints depend on."""

	model = Cherimoya(n_filters=32, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda().eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L, device='cuda')

	expected_profile, expected_counts = model(X)

	path = tmp_path / "model.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), device='cuda', compile=False).eval()

	with torch.no_grad():
		got_profile, got_counts = loaded(X)

	prof_diff = (expected_profile - got_profile).abs().max().item()
	count_diff = (expected_counts - got_counts).abs().max().item()
	assert prof_diff <= 1e-4, f"profile diverged after save/load: {prof_diff}"
	assert count_diff <= 1e-4, f"counts diverged after save/load: {count_diff}"


@pytest.mark.cuda
@pytest.mark.triton
def test_cherimoya_backward_matches_cpu_autograd():
	"""End-to-end gradient parity for the full Cherimoya model. CPU
	autograd uses pure PyTorch ops; CUDA autograd uses the Triton
	fwd+bwd kernels in every CheriBlock plus cuDNN/cuBLAS for the
	surrounding layers. The two must agree on every parameter grad and
	the input grad — this is the broadest regression guard for the
	training kernels.

	Tolerance budget: 3 stacked blocks, each with Triton conv+norm bwd
	feeding into cuBLAS linear bwd (TF32). Errors compound; allow
	~5e-3 absolute or relative. This is the realistic precision floor
	of fp32-with-TF32 training on GPU vs a fp32 CPU reference.

	The first GPU call to each block shape triggers Triton autotune,
	whose benchmark trials can contaminate the user-visible output via
	atomic_add residue in the bwd. We warm up to lock the configs
	before measuring."""

	torch.manual_seed(0)
	cpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False)
	gpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda()
	gpu_model.load_state_dict(cpu_model.state_dict())

	L = _input_window_for(cpu_model)

	# Warmup pass on GPU: triggers autotune for every block's fwd+bwd
	# kernels at the shapes this model uses. Each block has a different
	# dilation so each has its own autotune entry.
	with torch.enable_grad():
		xw = torch.randn(1, 4, L, device='cuda', requires_grad=True)
		yp, yc = gpu_model(xw)
		(yp.sum() + yc.sum()).backward()
		gpu_model.zero_grad()

	x_cpu = torch.randn(1, 4, L, requires_grad=True)
	x_gpu = x_cpu.detach().cuda().requires_grad_(True)

	y_prof_cpu, y_count_cpu = cpu_model(x_cpu)
	y_prof_gpu, y_count_gpu = gpu_model(x_gpu)

	(y_prof_cpu.sum() + y_count_cpu.sum()).backward()
	(y_prof_gpu.sum() + y_count_gpu.sum()).backward()

	# Forward outputs must agree first — otherwise grad comparison
	# isn't meaningful.
	assert torch.allclose(y_prof_cpu, y_prof_gpu.cpu(),
		atol=5e-3, rtol=5e-3), "forward profile diverged"
	assert torch.allclose(y_count_cpu, y_count_gpu.cpu(),
		atol=5e-3, rtol=5e-3), "forward counts diverged"

	# Per-parameter grad parity. Some params (lw0/lw1) aren't used in
	# forward — grad is None on both sides; skip them.
	cpu_params = dict(cpu_model.named_parameters())
	for name, gpu_p in gpu_model.named_parameters():
		cpu_p = cpu_params[name]
		if cpu_p.grad is None:
			assert gpu_p.grad is None, \
				f"{name}: CPU grad None but GPU grad is not"
			continue
		diff = (cpu_p.grad - gpu_p.grad.cpu()).abs().max().item()
		scale = max(cpu_p.grad.abs().max().item(), 1e-6)
		rel = diff / scale
		assert diff < 5e-3 or rel < 5e-3, \
			f"{name}: max-abs-diff={diff:.3e}  max-rel={rel:.3e}"

	# Input grad parity.
	in_diff = (x_cpu.grad - x_gpu.grad.cpu()).abs().max().item()
	assert in_diff < 5e-3, f"input grad max-abs-diff={in_diff:.3e}"


@pytest.mark.cuda
@pytest.mark.triton
def test_cherimoya_inference_megakernel_matches_cpu():
	"""End-to-end forward parity between the pure-PyTorch CPU model and
	the CUDA model under no_grad (which routes every CheriBlock through
	the new megakernel). bf16-dot precision accumulates across the
	stack of blocks, so the tolerance budget is looser than the
	single-block test but still pins a hard upper bound."""

	torch.manual_seed(0)
	cpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).eval()
	gpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda().eval()
	gpu_model.load_state_dict(cpu_model.state_dict())

	L = _input_window_for(cpu_model)
	x = torch.randn(1, 4, L)

	with torch.no_grad():
		y_prof_cpu, y_count_cpu = cpu_model(x)
		y_prof_gpu, y_count_gpu = gpu_model(x.cuda())

	prof_diff = (y_prof_cpu - y_prof_gpu.cpu()).abs().max().item()
	count_diff = (y_count_cpu - y_count_gpu.cpu()).abs().max().item()
	assert prof_diff <= 5e-2, \
		f"CPU vs CUDA-megakernel profile max-abs-diff={prof_diff:.3e}"
	assert count_diff <= 5e-2, \
		f"CPU vs CUDA-megakernel counts max-abs-diff={count_diff:.3e}"


@pytest.mark.cuda
@pytest.mark.triton
def test_model_no_grad_stable_across_repeated_calls_cuda():
	"""A weight cache keyed only on Parameter identity could silently
	stale-hit if the model is reused across many inference passes
	(e.g., in saturation mutagenesis loops). Verify deterministic output
	across repeated no_grad forwards on the same input."""

	model = Cherimoya(n_filters=32, n_layers=2, verbose=False).cuda().eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L, device='cuda')

	with torch.no_grad():
		p1, c1 = model(X)
		p2, c2 = model(X)
		p3, c3 = model(X.clone())

	assert torch.equal(p1, p2)
	assert torch.equal(c1, c2)
	assert torch.equal(p1, p3)
	assert torch.equal(c1, c3)


# --------- fit() measures / profile_jsd ------------------------------------

def _fit_tiny_model(tmp_path, signal_groups, measures, max_epochs=2, model=None):
	"""Smallest possible fit() run, parametrized over signal_groups and
	measures -- shared setup for the measures/profile_jsd tests below.
	Mirrors test_grouped_model_fit_smoke's synthetic-data harness. Pass
	an existing ``model`` to fit it a second time (e.g. to test that a
	repeat fit() call doesn't double-append Logger columns)."""

	import os
	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR
	from cherimoya.io import PeakNegativeSampler, channel_permutation_from_groups

	n_outputs = sum(signal_groups)
	if model is None:
		model = Cherimoya(n_filters=8, n_layers=2, signal_groups=signal_groups,
			verbose=False, compile=False)
	model.name = str(tmp_path / "smoke")

	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	g = torch.Generator().manual_seed(0)
	n_peaks, n_negs = 8, 4
	peak_sequences = torch.zeros(n_peaks, 4, L)
	peak_sequences[:, 0, :] = 1.0
	peak_signals = torch.randint(0, 5, (n_peaks, n_outputs, out_L),
		generator=g).float()
	neg_sequences = torch.zeros(n_negs, 4, L)
	neg_sequences[:, 0, :] = 1.0
	neg_signals = torch.zeros(n_negs, n_outputs, out_L)

	sampler = PeakNegativeSampler(
		peak_sequences=peak_sequences, peak_signals=peak_signals,
		negative_sequences=neg_sequences, negative_signals=neg_signals,
		in_window=L, out_window=out_L, max_jitter=0,
		negative_ratio=0, random_state=0, reverse_complement=True,
		signal_perm=channel_permutation_from_groups(signal_groups))
	training_data = torch.utils.data.DataLoader(sampler, batch_size=4,
		num_workers=0)

	muon_params, adam_params, lw_params = [], [], []
	for name, p in model.named_parameters():
		if name in ("lw0", "lw1"):
			lw_params.append(p)
		elif (p.ndim == 2 and "weight" in name and name != "linear.weight"
				and "conv_weight" not in name):
			muon_params.append(p)
		else:
			adam_params.append(p)
	muon_opt = Muon(muon_params, lr=1e-3, weight_decay=0.0)
	adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)
	lw_opt = torch.optim.SGD(lw_params, lr=1e-3, weight_decay=0.0, momentum=0.9)
	muon_sched = LinearLR(muon_opt, start_factor=1.0, total_iters=1)
	adam_sched = LinearLR(adam_opt, start_factor=1.0, total_iters=1)
	lw_sched = LinearLR(lw_opt, start_factor=1.0, total_iters=1)

	X_valid = torch.zeros(4, 4, L)
	X_valid[:, 0, :] = 1.0
	y_valid = torch.randint(0, 5, (4, n_outputs, out_L), generator=g).float()

	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		model.fit(training_data, muon_opt, adam_opt, lw_opt, muon_sched,
			adam_sched, lw_sched, X_valid=X_valid, X_ctl_valid=None,
			y_valid=y_valid, max_epochs=max_epochs, batch_size=4,
			dtype='float32', device='cpu', early_stopping=None,
			measures=measures)
	finally:
		os.chdir(cwd)

	return model


def test_default_measures_reproduces_current_schema(tmp_path):
	"""The default measures=('profile_pearson', 'count_pearson') must
	leave the Logger schema exactly as __init__ built it -- no JSD
	columns, zero behavior change from before this option existed."""

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		verbose=False, compile=False)
	original_summary_names = list(model.logger.names)
	original_detail_names = list(model.detail_logger.names)

	fitted = _fit_tiny_model(tmp_path, [1],
		measures=('profile_pearson', 'count_pearson'))

	assert fitted.logger.names == original_summary_names
	assert fitted.detail_logger.names == original_detail_names
	assert "Validation Profile JSD" not in fitted.logger.names
	assert not any(n.startswith("ProfileJSD_") for n in fitted.detail_logger.names)


def test_jsd_measure_adds_summary_and_detail_columns(tmp_path):
	model = _fit_tiny_model(tmp_path, [1, 1],
		measures=['profile_pearson', 'count_pearson', 'profile_jsd'])

	assert "Validation Profile JSD" in model.logger.names
	assert "ProfileJSD_g0" in model.detail_logger.names
	assert "ProfileJSD_g1" in model.detail_logger.names


def test_jsd_values_are_bounded(tmp_path):
	"""JSD is a bounded distance (sqrt of a JS divergence bounded by
	ln 2) -- values must land in [0, 1] regardless of input."""

	model = _fit_tiny_model(tmp_path, [1, 1],
		measures=['profile_pearson', 'count_pearson', 'profile_jsd'])

	summary_log = tmp_path / "smoke.log"
	detail_log = tmp_path / "smoke.detailed.log"
	import pandas
	summary_df = pandas.read_csv(summary_log, sep="\t")
	detail_df = pandas.read_csv(detail_log, sep="\t")

	assert summary_df["Validation Profile JSD"].between(0, 1).all()
	assert detail_df["ProfileJSD_g0"].between(0, 1).all()
	assert detail_df["ProfileJSD_g1"].between(0, 1).all()


@pytest.mark.parametrize("signal_groups", [[1], [1, 1], [1, 1, 1]])
def test_jsd_is_group_count_agnostic(tmp_path, signal_groups):
	"""Must produce exactly len(signal_groups) JSD columns, not a
	hardcoded count."""

	model = _fit_tiny_model(tmp_path, signal_groups,
		measures=['profile_pearson', 'count_pearson', 'profile_jsd'])

	jsd_columns = [n for n in model.detail_logger.names
		if n.startswith("ProfileJSD_g")]
	assert len(jsd_columns) == len(signal_groups)


def test_fit_can_be_called_twice_without_duplicating_jsd_columns(tmp_path):
	"""Logger schema is rebuilt from constants on every fit() call, not
	appended to whatever it currently is -- calling fit() a second time
	on the SAME model instance (e.g. continued training) must not double
	the JSD columns."""

	model = _fit_tiny_model(tmp_path, [1],
		measures=['profile_pearson', 'count_pearson', 'profile_jsd'],
		max_epochs=1)
	first_names = list(model.detail_logger.names)

	model = _fit_tiny_model(tmp_path, [1],
		measures=['profile_pearson', 'count_pearson', 'profile_jsd'],
		max_epochs=1, model=model)

	assert model.detail_logger.names == first_names
	assert model.detail_logger.names.count("ProfileJSD_g0") == 1
