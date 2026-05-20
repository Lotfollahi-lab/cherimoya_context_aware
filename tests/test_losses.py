"""Tests for the mixture loss."""

import pytest
import torch

from cherimoya.losses import _mixture_loss


def _toy_inputs(n=2, n_outputs=1, length=8, n_count_outputs=1, seed=0):
	g = torch.Generator().manual_seed(seed)
	y = torch.randint(0, 5, (n, n_outputs, length), generator=g).float()
	y_hat_logits = torch.randn(n, n_outputs, length, generator=g)
	y_hat_logcounts = torch.randn(n, n_count_outputs, generator=g)
	return y, y_hat_logits, y_hat_logcounts


def test_mixture_loss_returns_per_track_vectors():
	y, logits, logcounts = _toy_inputs()
	profile_loss, count_loss = _mixture_loss(y, logits, logcounts)
	assert profile_loss.shape == (1,)
	assert count_loss.shape == (1,)
	assert torch.isfinite(profile_loss).all()
	assert torch.isfinite(count_loss).all()


def test_mixture_loss_multi_track_shapes():
	y, logits, logcounts = _toy_inputs(n_outputs=3, n_count_outputs=3)
	profile_loss, count_loss = _mixture_loss(y, logits, logcounts)
	assert profile_loss.shape == (3,)
	assert count_loss.shape == (3,)


def test_mixture_loss_shared_count_head():
	# n_outputs=3 profile but a single shared count output: count loss
	# is computed against the total count across all tracks.
	y, logits, logcounts = _toy_inputs(n_outputs=3, n_count_outputs=1)
	profile_loss, count_loss = _mixture_loss(y, logits, logcounts)
	assert profile_loss.shape == (3,)
	assert count_loss.shape == (1,)


def test_mixture_loss_count_loss_zero_when_perfect_predictions():
	y = torch.full((1, 1, 4), 2.0)
	logits = torch.zeros(1, 1, 4)
	# log1p(sum(y)) where sum(y) = 8 -> log(9) ≈ 2.197
	true_logcounts = torch.tensor([[torch.log(torch.tensor(9.0))]])
	_, count_loss = _mixture_loss(y, logits, true_logcounts)
	assert torch.allclose(count_loss, torch.tensor(0.0), atol=1e-5)


def test_mixture_loss_labels_filter_excludes_negatives():
	"""When labels are provided, the profile loss is only computed over
	the labeled-1 examples, but the count loss runs on all examples."""

	y, logits, logcounts = _toy_inputs(n=4)
	labels = torch.tensor([1, 0, 1, 0])

	# All-positive subset should produce the same profile loss as the
	# full call when labels=None — we provide that as a baseline.
	y_pos = y[labels == 1]
	logits_pos = logits[labels == 1]
	logcounts_pos = logcounts[labels == 1]

	prof_with_labels, _ = _mixture_loss(y, logits, logcounts, labels=labels)
	prof_without_labels, _ = _mixture_loss(y_pos, logits_pos, logcounts_pos)

	assert torch.allclose(prof_with_labels, prof_without_labels, atol=1e-5)


def test_mixture_loss_count_loss_uses_all_examples_with_labels():
	"""The count loss should not be filtered by `labels`."""
	y, logits, logcounts = _toy_inputs(n=4)
	labels = torch.tensor([1, 0, 1, 0])

	_, count_with_labels = _mixture_loss(y, logits, logcounts, labels=labels)
	_, count_no_labels = _mixture_loss(y, logits, logcounts)
	assert torch.allclose(count_with_labels, count_no_labels, atol=1e-5)


def test_mixture_loss_is_differentiable():
	y, logits, logcounts = _toy_inputs()
	logits = logits.detach().requires_grad_(True)
	logcounts = logcounts.detach().requires_grad_(True)
	profile_loss, count_loss = _mixture_loss(y, logits, logcounts)
	(profile_loss + count_loss).backward()
	assert logits.grad is not None
	assert logcounts.grad is not None
	assert torch.isfinite(logits.grad).all()
	assert torch.isfinite(logcounts.grad).all()
