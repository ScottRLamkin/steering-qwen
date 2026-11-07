"""Shared parts for neuron-finding and neuron-editing methods, plus one recipe per paper.

* :mod:`.sites`       where a neuron is read (gate / act / up / down_in / down_out) and hooks there
* :mod:`.neuronset`   a scored set of neurons, saved as JSON with provenance
* :mod:`.collect`     per-group activation statistics over many texts
* :mod:`.stats`       firing-rate difference, Cohen's d, AUROC, average precision, entropy, probes
* :mod:`.select`      top-k, quantiles, caps, consensus, universal-neuron filter
* :mod:`.intervene`   one hook-based :class:`Intervention` for mul / set / add / clamp / patch edits
* :mod:`.attribution` attribution patching (AtP), integrated gradients, exact patching
* :mod:`.decoding`    DoLa contrastive layer decoding
* :mod:`.evaluate`    log-prob shifts, classifiers, lexicons, refusal rate, perplexity, controls
* :mod:`.data`        dataset loaders, cached under ``~/.cache/steering_llms``
* :mod:`.methods`     the recipes: ``discover(...) -> NeuronSet`` and ``edits(NeuronSet, ...)``
"""

from .collect import Collector, GroupStats, Sample, answer_sample, prompt_sample, text_sample
from .intervene import Edit, Intervention, edits_from_set, load_edits, save_edits
from .neuronset import NeuronSet, load_sets, save_sets
from .sites import Site, hook_site

__all__ = [
    "Collector",
    "Edit",
    "GroupStats",
    "Intervention",
    "NeuronSet",
    "Sample",
    "Site",
    "answer_sample",
    "edits_from_set",
    "hook_site",
    "load_edits",
    "load_sets",
    "prompt_sample",
    "save_edits",
    "save_sets",
    "text_sample",
]
