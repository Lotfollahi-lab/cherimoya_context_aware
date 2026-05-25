# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>


"""
This module contains the mixture loss function used for training Cherimoya
models, which is comprised of a multinomial log likelihood component and a
mean-squared error component. These losses are provided independently, so
other code can implement different ways of combining them into a single loss.
"""

import torch

from bpnetlite.losses import MNLLLoss

from .io import _validate_signal_groups


def _mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None,
		signal_groups=None):
	"""A function that takes in predictions and truth and returns the loss.

	This function takes in the observed integer read counts, the predicted logits,
	and the predicted logcounts, and returns per-*group* profile and count
	losses. Each channel is treated as an independent multinomial along the
	length dimension; the per-channel multinomial likelihoods are then
	averaged within each signal group so every group contributes one
	profile-loss term regardless of how many channels it contains. The
	count loss is per-group by construction (one prediction per group). When
	``signal_groups`` is None the function falls back to per-channel losses
	(every channel is its own group for the purposes of this aggregation),
	which matches the pre-grouping behavior.


	Parameters
	----------
	y: torch.Tensor, shape=(n, n_outputs, length)
		The observed counts for each example across each strand/output and at each
		position. This should likely be sparse integers.

	y_hat_logits: torch.Tensor, shape=(n, n_outputs, length)
		The predicted *logits* for each example across each strand/output and at
		each position. This will be normalized internally, so DO NOT run a softmax
		on your model.

	y_hat_logcounts: torch.Tensor, shape=(n, n_count_outputs)
		The predicted *log counts* for each example. ``n_count_outputs`` is
		``n_groups`` when ``signal_groups`` is given (per-group count head)
		or ``n_outputs`` otherwise (per-channel count head); the truth
		is derived from ``y`` accordingly.


	labels: torch.Tensor, shape=(n,), optional
		Whether the example is from a peak (1) or a non-peak (0). If provided, the
		profile loss will only be calculated on the peak examples. The count loss
		will always be calculated on the entire set of examples. If not provided,
		the profile loss will also be calculated on the entire set of examples.
		Default is None.

	signal_groups: list of int or None, optional
		Group sizes for the channel dimension of ``y``. When given, both
		the per-channel profile MNLLs and the true counts are pooled per
		group: a stranded ``(+, -)`` pair contributes one profile-loss
		term and one count-target. ``sum(signal_groups)`` must equal
		``y.shape[1]``. When None, every channel is treated as its own
		group (legacy behavior). Default is None.


	Returns
	-------
	profile_loss: torch.Tensor, shape=(n_groups,) or (n_outputs,)
		The per-group multinomial log likelihood (mean across the group's
		channels then mean across examples). Falls back to per-channel
		shape ``(n_outputs,)`` when ``signal_groups`` is None.

	count_loss: torch.Tensor, shape=(n_count_outputs,)
		The per-group (or per-track) mean-squared error on log(count+1),
		averaged across examples.
	"""

	log_probs = torch.nn.functional.log_softmax(y_hat_logits, dim=-1)

	# Per-channel true counts: (n, n_outputs).
	y_per_track = y.sum(dim=-1)

	if signal_groups is not None:
		_validate_signal_groups(signal_groups)
		if sum(signal_groups) != y_per_track.shape[-1]:
			raise ValueError(
				"sum(signal_groups)={} does not match y.shape[1]={}"
				.format(sum(signal_groups), y_per_track.shape[-1]))

	# Profile loss: per-example per-channel MNLL, then mean over examples
	# -> shape (n_outputs,).
	if labels is not None:
		mnll = MNLLLoss(log_probs[labels == 1], y[labels == 1])
	else:
		mnll = MNLLLoss(log_probs, y)
	profile_loss = mnll.mean(dim=0)

	# Pool the per-channel profile MNLL into a per-group mean so every
	# group contributes one loss term, regardless of channel count.
	# Pool the per-channel true counts likewise so the count target is
	# per-group. Both poolings are skipped for the all-size-one case
	# (pure identity) to avoid an unnecessary copy.
	if signal_groups is not None and len(signal_groups) != y_per_track.shape[-1]:
		groups_t = torch.tensor(signal_groups, device=y_per_track.device)
		group_idx = torch.repeat_interleave(
			torch.arange(len(signal_groups), device=y_per_track.device),
			groups_t)

		# Per-group profile loss: sum then divide by group size.
		profile_per_group = torch.zeros(len(signal_groups),
			device=profile_loss.device, dtype=profile_loss.dtype)
		profile_per_group.index_add_(0, group_idx, profile_loss)
		profile_per_group = profile_per_group / groups_t.to(
			profile_loss.dtype)
		profile_loss = profile_per_group

		# Per-group true counts: sum.
		n = y_per_track.shape[0]
		y_per_group = torch.zeros(n, len(signal_groups),
			device=y_per_track.device, dtype=y_per_track.dtype)
		y_per_group.index_add_(1, group_idx, y_per_track)
		y_per_track = y_per_group

	# Count loss: per-example per-group squared error, then mean over examples.
	count_sq_err = (torch.log(y_per_track + 1) - y_hat_logcounts) ** 2
	count_loss = count_sq_err.mean(dim=0)

	return profile_loss, count_loss
