"""Tests for cherimoya._wandb -- the optional wandb logging backend -- and
for Cherimoya.fit()'s wandb_config/signal_group_names wiring."""

import sys
from unittest import mock

import pytest

from cherimoya import _wandb


class _FakeRun:
	"""Minimal stand-in for a wandb.Run: records .log calls, tracks .finish."""

	def __init__(self):
		self.logged = []
		self.finished = False

	def log(self, payload, step=None):
		self.logged.append((step, payload))

	def finish(self):
		self.finished = True


# --------- log_epoch --------------------------------------------------------

def test_log_epoch_noop_when_run_is_none():
	# Must not raise even though the rest of the args are unused.
	_wandb.log_epoch(None, 0, ["Epoch"], [0], [], [])


def test_log_epoch_zips_column_names_with_row_values():
	"""The summary portion is schema-driven (zip), not a hardcoded list of
	metric keys -- so a future Logger column shows up automatically."""

	run = _FakeRun()
	_wandb.log_epoch(run, 3, ["Epoch", "Training MNLL"], [3, 1.5], [], [])
	assert len(run.logged) == 1
	step, payload = run.logged[0]
	assert step == 3
	assert payload["Epoch"] == 3
	assert payload["Training MNLL"] == 1.5


def test_log_epoch_logs_exactly_once_per_call():
	"""One wandb.log call per epoch, not one per metric -- avoids fighting
	wandb's own step counter."""

	run = _FakeRun()
	_wandb.log_epoch(run, 0, ["Epoch"], [0], [0.1, 0.2], [0.3, 0.4])
	assert len(run.logged) == 1


def test_log_epoch_uses_step_epoch():
	run = _FakeRun()
	_wandb.log_epoch(run, 7, ["Epoch"], [7], [], [])
	step, _ = run.logged[0]
	assert step == 7


def test_log_epoch_group_labels_default_to_generic_index():
	"""No signal_group_names -> group_{i}. Nothing dataset-specific."""

	run = _FakeRun()
	_wandb.log_epoch(run, 0, ["Epoch"], [0], [0.5, 0.6, 0.7], [0.1, 0.2, 0.3])
	_, payload = run.logged[0]
	assert payload["val/profile_pearson/group_0"] == 0.5
	assert payload["val/profile_pearson/group_1"] == 0.6
	assert payload["val/profile_pearson/group_2"] == 0.7
	assert payload["val/count_pearson/group_0"] == 0.1


def test_log_epoch_group_labels_use_signal_group_names():
	run = _FakeRun()
	_wandb.log_epoch(run, 0, ["Epoch"], [0], [0.5, 0.6], [0.1, 0.2],
		signal_group_names=["B_cell", "T_cell"])
	_, payload = run.logged[0]
	assert payload["val/profile_pearson/B_cell"] == 0.5
	assert payload["val/profile_pearson/T_cell"] == 0.6
	assert "val/profile_pearson/group_0" not in payload


@pytest.mark.parametrize("n_groups", [1, 2, 5])
def test_log_epoch_is_group_count_agnostic(n_groups):
	"""Must work for any number of signal groups, not a hardcoded dataset
	size (e.g. 21 BMMC celltypes) -- the fork is dataset-agnostic."""

	run = _FakeRun()
	profile = [float(i) for i in range(n_groups)]
	counts = [float(i) + 0.5 for i in range(n_groups)]
	_wandb.log_epoch(run, 0, ["Epoch"], [0], profile, counts)
	_, payload = run.logged[0]
	profile_keys = [k for k in payload if k.startswith("val/profile_pearson/")]
	count_keys = [k for k in payload if k.startswith("val/count_pearson/")]
	assert len(profile_keys) == n_groups
	assert len(count_keys) == n_groups


def test_log_epoch_soft_fails_on_run_log_exception():
	class _Boom:
		def log(self, *args, **kwargs):
			raise RuntimeError("network hiccup")

	_wandb.log_epoch(_Boom(), 0, ["Epoch"], [0], [], [])  # must not raise


def test_log_epoch_per_group_jsd_absent_by_default():
	"""per_group_jsd=None (the default) -> no val/profile_jsd/* keys at
	all, mirroring how the JSD summary/detail columns are absent unless
	requested."""

	run = _FakeRun()
	_wandb.log_epoch(run, 0, ["Epoch"], [0], [0.5], [0.1])
	_, payload = run.logged[0]
	assert not any(k.startswith("val/profile_jsd/") for k in payload)


def test_log_epoch_per_group_jsd_included_when_given():
	run = _FakeRun()
	_wandb.log_epoch(run, 0, ["Epoch"], [0], [0.5, 0.6], [0.1, 0.2],
		per_group_jsd=[0.2, 0.3])
	_, payload = run.logged[0]
	assert payload["val/profile_jsd/group_0"] == 0.2
	assert payload["val/profile_jsd/group_1"] == 0.3


def test_log_epoch_per_group_jsd_uses_signal_group_names():
	run = _FakeRun()
	_wandb.log_epoch(run, 0, ["Epoch"], [0], [0.5], [0.1],
		signal_group_names=["B_cell"], per_group_jsd=[0.2])
	_, payload = run.logged[0]
	assert payload["val/profile_jsd/B_cell"] == 0.2


# --------- init ---------------------------------------------------------------

def test_init_returns_none_when_wandb_not_installed(monkeypatch):
	monkeypatch.setitem(sys.modules, "wandb", None)
	assert _wandb.init({"project": "p"}) is None


def test_init_returns_none_without_api_key(monkeypatch):
	monkeypatch.delenv("WANDB_API_KEY", raising=False)
	monkeypatch.setattr(_wandb, "_api_key_from_netrc", lambda: None)
	assert _wandb.init({"project": "p"}) is None


def test_init_soft_fails_when_wandb_init_raises(monkeypatch):
	monkeypatch.setenv("WANDB_API_KEY", "fake-key")
	fake_wandb = mock.Mock()
	fake_wandb.init.side_effect = RuntimeError("boom")
	monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
	assert _wandb.init({"project": "p"}) is None


def test_init_calls_wandb_init_with_resolved_config(monkeypatch):
	monkeypatch.setenv("WANDB_API_KEY", "fake-key")
	fake_wandb = mock.Mock()
	fake_run = _FakeRun()
	fake_wandb.init.return_value = fake_run
	monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

	result = _wandb.init({"project": "myproj", "name": "run1", "tags": ["a"]})

	assert result is fake_run
	fake_wandb.init.assert_called_once()
	_, kwargs = fake_wandb.init.call_args
	assert kwargs["project"] == "myproj"
	assert kwargs["name"] == "run1"
	assert kwargs["tags"] == ["a"]


# --------- finish -------------------------------------------------------------

def test_finish_noop_when_run_is_none():
	_wandb.finish(None)  # must not raise


def test_finish_calls_run_finish():
	run = _FakeRun()
	_wandb.finish(run)
	assert run.finished


def test_finish_soft_fails_on_exception():
	class _Boom:
		def finish(self):
			raise RuntimeError("boom")

	_wandb.finish(_Boom())  # must not raise


# --------- Cherimoya.fit() wiring ----------------------------------------

def _tiny_fit_setup(tmp_path, signal_groups):
	"""Build the smallest possible fit() call, mirroring
	test_model.py::test_grouped_model_fit_smoke but parametrized over
	signal_groups so the wandb wiring can be checked group-count-agnostic."""

	import torch
	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR
	from cherimoya import Cherimoya
	from cherimoya.io import PeakNegativeSampler, channel_permutation_from_groups

	n_outputs = sum(signal_groups)
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=signal_groups,
		verbose=False, compile=False)
	model.name = str(tmp_path / "smoke")

	L = 2 * model.trimming + 64
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

	return dict(model=model, training_data=training_data,
		muon_optimizer=muon_opt, adam_optimizer=adam_opt, lw_optimizer=lw_opt,
		muon_scheduler=muon_sched, adam_scheduler=adam_sched,
		lw_scheduler=lw_sched, X_valid=X_valid, X_ctl_valid=None,
		y_valid=y_valid)


def test_fit_wandb_absent_makes_zero_wandb_calls(tmp_path, monkeypatch):
	"""wandb_config=None (the default) -> the real wandb package is never
	touched, and disk logs are written exactly as before."""

	import os

	fake_wandb = mock.Mock()
	monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

	setup = _tiny_fit_setup(tmp_path, [1])
	model = setup.pop("model")

	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		best = model.fit(setup["training_data"], setup["muon_optimizer"],
			setup["adam_optimizer"], setup["lw_optimizer"],
			setup["muon_scheduler"], setup["adam_scheduler"],
			setup["lw_scheduler"], X_valid=setup["X_valid"],
			X_ctl_valid=setup["X_ctl_valid"], y_valid=setup["y_valid"],
			max_epochs=2, batch_size=4, dtype='float32', device='cpu',
			early_stopping=None, wandb_config=None)
	finally:
		os.chdir(cwd)

	import math
	assert math.isfinite(float(best))
	fake_wandb.init.assert_not_called()
	fake_wandb.log.assert_not_called()

	assert (tmp_path / "smoke.log").exists()
	assert (tmp_path / "smoke.detailed.log").exists()


@pytest.mark.parametrize("signal_groups", [[1], [1, 1], [1, 1, 1]])
def test_fit_wandb_configured_logs_once_per_epoch(tmp_path, monkeypatch,
		signal_groups):
	"""wandb_config given -> one wandb_run.log(..., step=epoch) call per
	epoch, with N per-group entries where N == len(signal_groups) --
	never a hardcoded group count."""

	import os

	fake_run = _FakeRun()
	monkeypatch.setattr(_wandb, "init", lambda cfg: fake_run)

	setup = _tiny_fit_setup(tmp_path, signal_groups)
	model = setup.pop("model")

	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		model.fit(setup["training_data"], setup["muon_optimizer"],
			setup["adam_optimizer"], setup["lw_optimizer"],
			setup["muon_scheduler"], setup["adam_scheduler"],
			setup["lw_scheduler"], X_valid=setup["X_valid"],
			X_ctl_valid=setup["X_ctl_valid"], y_valid=setup["y_valid"],
			max_epochs=2, batch_size=4, dtype='float32', device='cpu',
			early_stopping=None, wandb_config={"project": "p"},
			signal_group_names=None)
	finally:
		os.chdir(cwd)

	assert len(fake_run.logged) == 2
	steps = [step for step, _ in fake_run.logged]
	assert steps == [0, 1]

	for _, payload in fake_run.logged:
		assert payload["Epoch"] in (0, 1)
		profile_keys = [k for k in payload
			if k.startswith("val/profile_pearson/group_")]
		count_keys = [k for k in payload
			if k.startswith("val/count_pearson/group_")]
		assert len(profile_keys) == len(signal_groups)
		assert len(count_keys) == len(signal_groups)

	assert fake_run.finished


def test_fit_wandb_uses_signal_group_names(tmp_path, monkeypatch):
	fake_run = _FakeRun()
	monkeypatch.setattr(_wandb, "init", lambda cfg: fake_run)

	setup = _tiny_fit_setup(tmp_path, [1, 1])
	model = setup.pop("model")

	import os
	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		model.fit(setup["training_data"], setup["muon_optimizer"],
			setup["adam_optimizer"], setup["lw_optimizer"],
			setup["muon_scheduler"], setup["adam_scheduler"],
			setup["lw_scheduler"], X_valid=setup["X_valid"],
			X_ctl_valid=setup["X_ctl_valid"], y_valid=setup["y_valid"],
			max_epochs=1, batch_size=4, dtype='float32', device='cpu',
			early_stopping=None, wandb_config={"project": "p"},
			signal_group_names=["alpha", "beta"])
	finally:
		os.chdir(cwd)

	_, payload = fake_run.logged[0]
	assert "val/profile_pearson/alpha" in payload
	assert "val/profile_pearson/beta" in payload
	assert "val/profile_pearson/group_0" not in payload


def test_fit_wandb_jsd_absent_when_not_requested(tmp_path, monkeypatch):
	"""Default measures (no 'profile_jsd') -> no val/profile_jsd/* keys
	reach wandb, end to end through the real fit() call -- not just the
	log_epoch unit tests above."""

	fake_run = _FakeRun()
	monkeypatch.setattr(_wandb, "init", lambda cfg: fake_run)

	setup = _tiny_fit_setup(tmp_path, [1])
	model = setup.pop("model")

	import os
	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		model.fit(setup["training_data"], setup["muon_optimizer"],
			setup["adam_optimizer"], setup["lw_optimizer"],
			setup["muon_scheduler"], setup["adam_scheduler"],
			setup["lw_scheduler"], X_valid=setup["X_valid"],
			X_ctl_valid=setup["X_ctl_valid"], y_valid=setup["y_valid"],
			max_epochs=1, batch_size=4, dtype='float32', device='cpu',
			early_stopping=None, wandb_config={"project": "p"})
	finally:
		os.chdir(cwd)

	_, payload = fake_run.logged[0]
	assert not any(k.startswith("val/profile_jsd/") for k in payload)
	assert "Validation Profile JSD" not in payload


def test_fit_wandb_jsd_included_when_requested(tmp_path, monkeypatch):
	"""measures including 'profile_jsd' -> the summary JSD-mean column
	(via the schema-zip, zero _wandb.py awareness needed) AND the
	per-group val/profile_jsd/* keys (via the explicit per_group_jsd
	param) both reach the wandb payload."""

	fake_run = _FakeRun()
	monkeypatch.setattr(_wandb, "init", lambda cfg: fake_run)

	setup = _tiny_fit_setup(tmp_path, [1, 1])
	model = setup.pop("model")

	import os
	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		model.fit(setup["training_data"], setup["muon_optimizer"],
			setup["adam_optimizer"], setup["lw_optimizer"],
			setup["muon_scheduler"], setup["adam_scheduler"],
			setup["lw_scheduler"], X_valid=setup["X_valid"],
			X_ctl_valid=setup["X_ctl_valid"], y_valid=setup["y_valid"],
			max_epochs=1, batch_size=4, dtype='float32', device='cpu',
			early_stopping=None, wandb_config={"project": "p"},
			measures=['profile_pearson', 'count_pearson', 'profile_jsd'])
	finally:
		os.chdir(cwd)

	_, payload = fake_run.logged[0]
	assert "Validation Profile JSD" in payload
	assert "val/profile_jsd/group_0" in payload
	assert "val/profile_jsd/group_1" in payload


def test_fit_wandb_finish_called_even_on_exception(tmp_path, monkeypatch):
	"""The finally block must call wandb_run.finish() even if the epoch
	loop raises."""

	import os
	fake_run = _FakeRun()
	monkeypatch.setattr(_wandb, "init", lambda cfg: fake_run)

	setup = _tiny_fit_setup(tmp_path, [1])
	model = setup.pop("model")

	def _boom(*args, **kwargs):
		raise RuntimeError("simulated training failure")

	monkeypatch.setattr(_wandb, "log_epoch", _boom)

	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		with pytest.raises(RuntimeError, match="simulated training failure"):
			model.fit(setup["training_data"], setup["muon_optimizer"],
				setup["adam_optimizer"], setup["lw_optimizer"],
				setup["muon_scheduler"], setup["adam_scheduler"],
				setup["lw_scheduler"], X_valid=setup["X_valid"],
				X_ctl_valid=setup["X_ctl_valid"], y_valid=setup["y_valid"],
				max_epochs=2, batch_size=4, dtype='float32', device='cpu',
				early_stopping=None, wandb_config={"project": "p"})
	finally:
		os.chdir(cwd)

	assert fake_run.finished
