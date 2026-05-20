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


def _mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None):
	"""A function that takes in predictions and truth and returns the loss.

	This function takes in the observed integer read counts, the predicted logits,
	and the predicted logcounts, and returns per-track profile and count losses.
	Each track is treated as an independent multinomial along the length
	dimension, and the count loss is computed per-track unless the count head
	produces a single shared output (in which case it is computed on the total
	count across tracks).


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
		either 1 (shared count head) or ``n_outputs`` (per-track count head);
		the truth is derived from `y` accordingly.


	labels: torch.Tensor, shape=(n,), optional
		Whether the example is from a peak (1) or a non-peak (0). If provided, the
		profile loss will only be calculated on the peak examples. The count loss
		will always be calculated on the entire set of examples. If not provided,
		the profile loss will also be calculated on the entire set of examples.
		Default is None.


	Returns
	-------
	profile_loss: torch.Tensor, shape=(n_outputs,)
		The per-track multinomial log likelihood, averaged across examples.

	count_loss: torch.Tensor, shape=(n_count_outputs,)
		The per-track mean-squared error on log(count+1), averaged across
		examples.
	"""

	log_probs = torch.nn.functional.log_softmax(y_hat_logits, dim=-1)

	# Per-track true counts: (n, n_outputs)
	y_per_track = y.sum(dim=-1)

	# Match the count head: collapse across tracks if the head is shared.
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

	# Count loss: per-example per-track squared error, then mean over examples.
	count_sq_err = (torch.log(y_ + 1) - y_hat_logcounts) ** 2
	count_loss = count_sq_err.mean(dim=0)

	return profile_loss, count_loss
