# _wandb.py

"""Optional wandb logging backend for ``Cherimoya.fit()``.

Cherimoya's own ``Logger`` (``bpnetlite.logging.Logger``) writes rows
straight to disk at the end of every epoch with no callback hook -- see
``fit()``'s per-epoch block in ``cherimoya.py``. This module is called
from that same block, synchronously, so wandb gets each epoch's metrics
the moment they're written to disk. No thread, no polling, no file
reads.

Every function here is soft-fail: wandb not being installed, no API key
being available, or any call into the ``wandb`` package raising, all
degrade to "logging didn't happen" rather than interrupting training.
``wandb`` is imported lazily inside each function so it stays a fully
optional dependency -- nothing in this module runs unless a caller
opts in with a ``wandb_config``.

This module knows nothing about any particular dataset. Per-group
labels come from whatever ``signal_group_names`` the caller passes in;
absent that, groups are labeled generically (``group_0``, ``group_1``,
...).
"""

import os


def _api_key_from_netrc():
	"""``WANDB_API_KEY`` fallback: the ``api.wandb.ai`` machine entry in
	``~/.netrc``, read with the stdlib ``netrc`` module. Returns None if
	there's no file, no entry, or the file doesn't parse -- never raises.
	"""

	import netrc

	path = os.path.expanduser("~/.netrc")
	if not os.path.exists(path):
		return None
	try:
		auth = netrc.netrc(path).authenticators("api.wandb.ai")
	except netrc.NetrcParseError:
		return None
	return auth[2] if auth else None


def init(wandb_config):
	"""Start a wandb run, or return None if that's not possible.

	Parameters
	----------
	wandb_config: dict
		``project``, ``name``, ``entity``, ``tags``, ``mode``, and
		``config`` are all optional; missing keys are passed to
		``wandb.init`` as None (wandb's own defaults apply).

	Returns
	-------
	run: wandb.Run or None
		None if wandb isn't installed, no API key is available (env var
		``WANDB_API_KEY`` if set, else ``~/.netrc``'s ``api.wandb.ai``
		entry), or ``wandb.init`` itself raises for any reason.
	"""

	try:
		import wandb
	except ImportError:
		print("wandb not available; continuing without it (disk logs "
			"unaffected).")
		return None

	if not os.environ.get("WANDB_API_KEY"):
		key = _api_key_from_netrc()
		if key:
			os.environ["WANDB_API_KEY"] = key

	if not os.environ.get("WANDB_API_KEY"):
		print("No WANDB_API_KEY and no ~/.netrc api.wandb.ai entry; "
			"continuing without wandb (disk logs unaffected).")
		return None

	try:
		return wandb.init(
			project=wandb_config.get("project"),
			name=wandb_config.get("name"),
			entity=wandb_config.get("entity"),
			tags=wandb_config.get("tags"),
			mode=wandb_config.get("mode"),
			config=wandb_config.get("config"),
		)
	except Exception as exc:  # noqa: BLE001 -- any wandb failure must degrade, not crash
		print("wandb.init failed ({}); continuing without wandb (disk "
			"logs unaffected).".format(exc))
		return None


def log_epoch(run, epoch, column_names, row_values, per_group_profile,
		per_group_count, signal_group_names=None):
	"""Log one epoch's metrics as a single ``wandb.log`` call.

	The summary portion is built by zipping ``column_names`` with
	``row_values`` -- the same names/row Cherimoya's own summary logger
	is given -- so this picks up any future summary column Cherimoya
	adds without needing an edit here. Per-group series are added
	separately, labeled by ``signal_group_names[i]`` if given, else
	``"group_{i}"`` -- this is the only dataset-specific piece, and it's
	optional.

	Parameters
	----------
	run: wandb.Run or None
		The run returned by :func:`init`. If None, this is a no-op.

	epoch: int
		The current epoch, used as wandb's ``step``.

	column_names: list of str
		Names for ``row_values`` (Cherimoya's summary logger schema).

	row_values: list
		One epoch's summary row, same length/order as ``column_names``.

	per_group_profile: list of float
		Per-group validation profile Pearson, one value per signal group.

	per_group_count: list of float
		Per-group validation count Pearson, one value per signal group.

	signal_group_names: list of str or None, optional
		Human-readable label per signal group, same length as
		``per_group_profile``/``per_group_count``. Default is None,
		which labels groups ``group_0``, ``group_1``, ....
	"""

	if run is None:
		return

	try:
		payload = dict(zip(column_names, row_values))

		for i, value in enumerate(per_group_profile):
			label = signal_group_names[i] if signal_group_names else "group_{}".format(i)
			payload["val/profile_pearson/{}".format(label)] = value

		for i, value in enumerate(per_group_count):
			label = signal_group_names[i] if signal_group_names else "group_{}".format(i)
			payload["val/count_pearson/{}".format(label)] = value

		run.log(payload, step=epoch)
	except Exception as exc:  # noqa: BLE001 -- a dropped metric must never kill training
		print("wandb log failed (epoch {}): {} -- continuing without it.".format(
			epoch, exc))


def finish(run):
	"""End a wandb run, soft-failing like everything else in this module."""

	if run is None:
		return

	try:
		run.finish()
	except Exception as exc:  # noqa: BLE001 -- must never mask the real exit path
		print("wandb.finish failed ({}); continuing.".format(exc))
