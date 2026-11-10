# Reproducible figure recipe

`williamson5_t63_overview.py` renders the root README's Williamson-5 figure
from the committed saved run using a pinned plotting recipe. It writes the
run's derived `assets/` and sidecar; it does not integrate or change primary
evidence. Physical reconstruction requires CUDA. Read the script before
running it and preserve its input hashes.

[Published runs](../runs/README.md) documents the capsule and checksum
receipt; the [Williamson report](../validation/williamson5_mri_2026-07-30.md)
interprets the numerical comparison. Other illustration recipes remain with
their [validation records](../validation/README.md), or use
[`readme_figures.py`](../readme_figures.py) and [asset provenance](../assets/README.md).
Held–Suarez rendering uses the [saved-run climate API](../../src/tropoi/representation/visual/HELD_SUAREZ_CLIMATE.md).
Large local/external inputs are not all bundled. Regenerate from the stated
inputs instead of hand-editing a figure or receipt.
