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


def _mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None,
		signal_groups=None):
	"""A function that takes in predictions and truth and returns the loss.

	This function takes in the observed integer read counts, the predicted logits,
	and the predicted logcounts, and returns per-track profile and per-group
	count losses. Each channel is treated as an independent multinomial along
	the length dimension. The count loss is computed per-group when
	``signal_groups`` is given (one count per modality — e.g. one count for
	an unstranded ATAC track, one count for a stranded ``(+, -)`` pair), per-
	channel otherwise (legacy), and on the total across all channels when the
	count head produces a single shared output.


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
		either 1 (shared count head), ``n_groups`` (per-group count head
		when ``signal_groups`` is given), or ``n_outputs`` (per-channel
		count head, legacy); the truth is derived from ``y`` accordingly.


	labels: torch.Tensor, shape=(n,), optional
		Whether the example is from a peak (1) or a non-peak (0). If provided, the
		profile loss will only be calculated on the peak examples. The count loss
		will always be calculated on the entire set of examples. If not provided,
		the profile loss will also be calculated on the entire set of examples.
		Default is None.

	signal_groups: list of int or None, optional
		Group sizes for the channel dimension of ``y``. When given, the
		true counts are pooled per group before being compared against
		``y_hat_logcounts``, so a stranded ``(+, -)`` pair contributes a
		single count target. ``sum(signal_groups)`` must equal
		``y.shape[1]``. When None, every channel is treated as its own
		group (legacy behavior). Default is None.


	Returns
	-------
	profile_loss: torch.Tensor, shape=(n_outputs,)
		The per-track multinomial log likelihood, averaged across examples.

	count_loss: torch.Tensor, shape=(n_count_outputs,)
		The per-group (or per-track) mean-squared error on log(count+1),
		averaged across examples.
	"""

	log_probs = torch.nn.functional.log_softmax(y_hat_logits, dim=-1)

	# Per-channel true counts: (n, n_outputs)
	y_per_track = y.sum(dim=-1)

	# Pool per-channel counts into per-group counts when grouping is
	# given. The stranded ``(+, -)`` pair sums its two strands into a
	# single per-group count, matching the per-group count head.
	if signal_groups is not None:
		if sum(signal_groups) != y_per_track.shape[-1]:
			raise ValueError(
				"sum(signal_groups)={} does not match y.shape[1]={}"
				.format(sum(signal_groups), y_per_track.shape[-1]))
		if len(signal_groups) == y_per_track.shape[-1]:
			# All groups are size 1 — pooling is a no-op, skip the
			# index_add to avoid an unnecessary copy.
			y_per_group = y_per_track
		else:
			n = y_per_track.shape[0]
			group_idx = torch.repeat_interleave(
				torch.arange(len(signal_groups), device=y_per_track.device),
				torch.tensor(signal_groups, device=y_per_track.device))
			y_per_group = torch.zeros(n, len(signal_groups),
				device=y_per_track.device, dtype=y_per_track.dtype)
			y_per_group.index_add_(1, group_idx, y_per_track)
		y_per_track = y_per_group

	# Match the count head: collapse across tracks/groups if the head is shared.
	if y_hat_logcounts.shape[-1] == 1 and y_per_track.shape[-1] > 1:
		y_ = y_per_track.sum(dim=-1, keepdim=True)
	else:
		y_ = y_per_track

	# Profile loss: per-example per-track MNLL, then mean over examples.
	if labels is not None:
		mnll = MNLLLoss(log_probs[labels == 1], y[labels == 1])
	else:
		mnll = MNLLLoss(log_probs, y)
	profile_loss = mnll.mean(dim=0)

	# Count loss: per-example per-group squared error, then mean over examples.
	count_sq_err = (torch.log(y_ + 1) - y_hat_logcounts) ** 2
	count_loss = count_sq_err.mean(dim=0)

	return profile_loss, count_loss
