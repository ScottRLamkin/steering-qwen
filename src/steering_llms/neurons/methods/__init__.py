"""One recipe per paper. Each module defines:

* ``NAME``, ``HELP``
* ``add_args(parser)``                 its knobs, with defaults from the paper's code
* ``discover(ctx, args) -> {key: NeuronSet}``
* ``edits(sets, key, direction, ...)`` the paper's intervention as a list of :class:`Edit`
* ``evaluate(ctx, sets, args) -> dict``

Modules are imported on first use so ``neurons --help`` stays fast.
"""

import importlib
from types import ModuleType

MODULES = {
    "hedwig": "hedwig",
    "dpn-le": "dpn_le",
    "pcn": "pcn",
    "npti": "npti",
    "iris": "iris",
    "sneuron-tst": "sneuron_tst",
    "aura": "aura",
    "gender": "gender",
    "cps": "cps",
    "knowledge": "knowledge",
    "atp": "atp",
    "sentiment": "sentiment",
    "emotion": "emotion",
    "safety": "safety",
}

PERSONALITY = ["hedwig", "dpn-le", "pcn", "npti", "iris"]


def get(name: str) -> ModuleType:
    if name not in MODULES:
        raise KeyError(f"unknown method {name!r}; one of {', '.join(MODULES)}")
    return importlib.import_module(f".{MODULES[name]}", __name__)
