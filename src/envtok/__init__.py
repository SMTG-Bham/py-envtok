"""Local-environment tokens and coverage-guaranteed subsets with py-envtok.

Typical use::

    from envtok import Corpus, CoverageBuilder, Selector, Tokenizer

    corpus = Corpus.from_source("omat24.yaml")
    cache = Tokenizer(profile="reference-723696d").run(
        corpus, "runs/omat24", workers=16
    )
    classes = CoverageBuilder(min_structure_support=100).fit(cache, split="train")
    result = Selector(mode="cover", cost="atoms").select(classes)
    result.validate()
    result.write_manifest("runs/omat24/selection")
    result.export("runs/omat24/selection/subset.extxyz", format="extxyz")
"""

from importlib.metadata import version

from envtok.coarsen import CoverageBuilder, CoverageClasses
from envtok.corpus import Corpus
from envtok.execution import ExecutionSettings
from envtok.profiles import get_profile, with_net_charge
from envtok.selection import SelectionResult, Selector
from envtok.store.cache import TokenCache
from envtok.tokenization import Tokenizer
from envtok.tokenizer import SiteToken

__version__ = version("py-envtok")

__all__ = [
    "Corpus",
    "CoverageBuilder",
    "CoverageClasses",
    "ExecutionSettings",
    "SelectionResult",
    "Selector",
    "SiteToken",
    "TokenCache",
    "Tokenizer",
    "__version__",
    "get_profile",
    "with_net_charge",
]
