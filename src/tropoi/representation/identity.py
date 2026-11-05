"""Scientific identity of a run configuration (stdlib only).

One rule, in one place: which configuration keys take part in a run's
*scientific* identity. Two consumers need exactly the same answer and must
never drift apart:

* the configuration layer, when it reports a resolved run's scientific
  subset (``*RunConfig.scientific_config_dict``);
* the archive, when it hashes that subset into a run id
  (:func:`tropoi.representation.archive.writer._config_hash`) — including
  when it recomputes the hash of a *saved* capsule's ``config.json``, where
  no configuration object exists any more.

The rule therefore lives in the representation layer, below the
configuration layer that produces run configurations and beside the archive
that persists and re-reads them. It is deliberately outside
``representation.archive``: importing it must stay free, so that a
configuration module does not pull the capsule reader (and NumPy) in.
"""
from __future__ import annotations

from typing import Mapping

#: Config keys excluded from the run-id hash: purely locational/control
#: values and derived artifacts. Single source of truth shared by
#: ``*RunConfig.scientific_config_dict`` and the archive's ``_config_hash``
#: so the hash exclusion logic is never duplicated (and can never drift).
#: ``initial_state_label`` (custom runs only) is a descriptive name; the
#: state's identity is its ``initial_state_digest``, which IS hashed.
HASH_EXCLUDED_KEYS: tuple[str, ...] = ("out", "experiment", "overwrite", "plots",
                                       "initial_state_label")


def scientific_config_subset(config: Mapping) -> dict:
    """Config subset that defines a run's *scientific* identity.

    Drops purely locational/control values (``out``, ``experiment``,
    ``overwrite``) and derived artifacts (``plots``) so two runs of the same
    science hash identically regardless of output location or plot selection.
    """
    return {k: v for k, v in dict(config).items() if k not in HASH_EXCLUDED_KEYS}
