"""Tests for the Cherimoya model construction and forward pass."""

import os

import pytest
import torch

from cherimoya import Cherimoya


@pytest.fixture
def small_model_kwargs():
	# A tiny but valid configuration that runs quickly on CPU.
	return dict(n_filters=8, n_layers=3, n_outputs=1, n_control_tracks=0,
		single_count_output=True, verbose=False)


def _input_window_for(model):
	"""Compute a valid input window length for `model`."""
	# Output window must be > 0; trimming bytes are removed from each side.
	return 2 * model.trimming + 64


# --------- Construction ----------------------------------------------------

def test_default_construction():
	model = Cherimoya(verbose=False)
	assert model.n_filters == 96
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
	loaded = Cherimoya.load(str(path))
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
	loaded = Cherimoya.load(str(path))
	assert loaded.expansion == 4
	for block in loaded.blocks:
		assert block.linear1.out_features == 32


def test_custom_construction(small_model_kwargs):
	model = Cherimoya(**small_model_kwargs)
	assert model.n_filters == 8
	assert model.n_layers == 3
	assert len(model.blocks) == 3


def test_n_outputs_and_control_tracks_shape():
	model = Cherimoya(n_filters=8, n_layers=2, n_outputs=2,
		n_control_tracks=2, single_count_output=False, verbose=False)
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


def test_signal_groups_legacy_n_outputs_maps_to_all_unstranded():
	"""Pre-grouping callers passed only n_outputs. That must now be
	interpreted as N independent unstranded groups (matching the new
	flat-list convention in the data pipeline)."""

	model = Cherimoya(n_filters=8, n_layers=2, n_outputs=3, verbose=False)
	assert model.signal_groups == [1, 1, 1]
	assert model.n_outputs == 3
	assert model.n_groups == 3


def test_signal_groups_grouped_pair():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		single_count_output=False, verbose=False)
	assert model.signal_groups == [1, 2]
	assert model.n_outputs == 3   # 1 + 2 channels total
	assert model.n_groups == 2    # 2 groups
	assert model.fconv.out_channels == 3
	# Per-group count head when single_count_output=False.
	assert model.linear.out_features == 2
	assert model.lw0.shape == (3,)
	assert model.lw1.shape == (2,)


def test_signal_groups_grouped_pair_with_shared_count_head():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		single_count_output=True, verbose=False)
	# Shared count head: still one output regardless of grouping.
	assert model.linear.out_features == 1
	assert model.lw1.shape == (1,)


def test_signal_groups_mismatch_with_n_outputs_raises():
	with pytest.raises(ValueError, match="disagrees with sum"):
		Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
			n_outputs=4, verbose=False)


def test_signal_groups_round_trips_through_save_load(tmp_path):
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		single_count_output=False, verbose=False)
	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.signal_groups == [1, 2]
	assert loaded.n_outputs == 3
	assert loaded.n_groups == 2


def test_load_legacy_checkpoint_with_only_n_outputs(tmp_path):
	"""A checkpoint saved with the pre-grouping format (config has
	n_outputs but no signal_groups) must load and be reinterpreted as
	all-unstranded groups."""

	# Hand-craft a legacy-style checkpoint without signal_groups.
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 1, 1],
		single_count_output=False, verbose=False)
	state = model.state_dict()
	legacy_config = {
		'n_filters': 8, 'n_layers': 2, 'n_outputs': 3,
		'n_control_tracks': 0, 'expansion': 2, 'residual_scale': 0.15,
		'name': 'cherimoya.8.2', 'trimming': model.trimming,
		'single_count_output': False, 'verbose': False,
	}
	path = tmp_path / "legacy.torch"
	torch.save({'config': legacy_config, 'state_dict': state}, str(path))

	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.signal_groups == [1, 1, 1]
	assert loaded.n_outputs == 3


@pytest.mark.parametrize("n_outputs,single_count_output,expected_lw1", [
	(1, True,  (1,)),   # single-task: back-compat with pre-vector checkpoints
	(1, False, (1,)),   # single-task, per-track count head: same shape
	(3, True,  (1,)),   # multi-track profile, shared count head
	(3, False, (3,)),   # multi-track profile, per-track count head
])
def test_lw0_lw1_shapes(n_outputs, single_count_output, expected_lw1):
	"""lw0 sizes with n_outputs (one weight per profile track); lw1 sizes
	with the count-head output dim (1 when single_count_output=True else
	n_outputs)."""

	model = Cherimoya(n_filters=8, n_layers=2, n_outputs=n_outputs,
		single_count_output=single_count_output, verbose=False)
	assert model.lw0.shape == (n_outputs,)
	assert model.lw1.shape == expected_lw1
	# Both initialize to ones; the fit loop relies on this.
	assert torch.allclose(model.lw0, torch.ones(n_outputs))
	assert torch.allclose(model.lw1, torch.ones(*expected_lw1))


@pytest.mark.parametrize("n_outputs,single_count_output", [
	(1, True),    # the format every existing checkpoint was saved in
	(3, False),
	(3, True),
])
def test_lw0_lw1_save_load_round_trip(tmp_path, n_outputs, single_count_output):
	"""Save/load preserves lw0/lw1 shapes and values across all
	n_outputs / single_count_output combinations. The (1, True) case
	also exercises back-compat with checkpoints saved before lw0/lw1
	became vectors (where the state_dict already stored shape (1,))."""

	model = Cherimoya(n_filters=8, n_layers=2, n_outputs=n_outputs,
		single_count_output=single_count_output, verbose=False)
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
	model = Cherimoya(n_filters=8, n_layers=2, n_outputs=1,
		n_control_tracks=2, single_count_output=True, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	X_ctl = torch.randn(1, 2, L)
	y_profile, y_counts = model(X, X_ctl)
	assert y_profile.shape == (1, 1, L - 2 * model.trimming)
	assert y_counts.shape == (1, 1)


def test_forward_multi_output_per_track_counts():
	model = Cherimoya(n_filters=8, n_layers=2, n_outputs=3,
		n_control_tracks=0, single_count_output=False, verbose=False).eval()
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

	loaded = Cherimoya.load(str(path)).eval()
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
	loaded = Cherimoya.load(str(path), device=device)
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
	loaded = Cherimoya.load(str(path)).eval()

	with torch.no_grad():
		got_profile, got_counts = loaded(X)

	assert torch.allclose(expected_profile, got_profile, atol=1e-6)
	assert torch.allclose(expected_counts, got_counts, atol=1e-6)


def test_model_no_grad_matches_grad_with_controls():
	"""Same parity check, exercising the control-tracks branch of the
	forward where the inference path's residual layout could in
	principle differ."""

	model = Cherimoya(n_filters=8, n_layers=2, n_outputs=1,
		n_control_tracks=2, single_count_output=True, verbose=False).eval()
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

	model = Cherimoya(n_filters=8, n_layers=2, expansion=1, n_outputs=1,
		n_control_tracks=0, single_count_output=True, verbose=False).eval()
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

	model = Cherimoya(n_filters=32, n_layers=3, n_outputs=1,
		n_control_tracks=0, single_count_output=True,
		verbose=False).cuda().eval()
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

	model = Cherimoya(n_filters=32, n_layers=3, n_outputs=1,
		n_control_tracks=0, single_count_output=True,
		verbose=False).cuda().eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L, device='cuda')

	expected_profile, expected_counts = model(X)

	path = tmp_path / "model.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), device='cuda').eval()

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
	cpu_model = Cherimoya(n_filters=16, n_layers=3, n_outputs=1,
		n_control_tracks=0, single_count_output=True, verbose=False)
	gpu_model = Cherimoya(n_filters=16, n_layers=3, n_outputs=1,
		n_control_tracks=0, single_count_output=True, verbose=False).cuda()
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
	cpu_model = Cherimoya(n_filters=16, n_layers=3, n_outputs=1,
		n_control_tracks=0, single_count_output=True,
		verbose=False).eval()
	gpu_model = Cherimoya(n_filters=16, n_layers=3, n_outputs=1,
		n_control_tracks=0, single_count_output=True,
		verbose=False).cuda().eval()
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
