"""
Load a bechmark loader, given the benchmark name.
"""
from functools import partial

from evals.mcqs.benchmarks.arc import load_arc
from evals.mcqs.benchmarks.blimp import load_blimp
from evals.mcqs.benchmarks.hellaswag import load_hellaswag
from evals.mcqs.benchmarks.mmlu import load_mmlu
from evals.mcqs.benchmarks.winogrande import load_winogrande
from evals.mcqs.benchmarks.langgame import load_langgame

EVALS_DICT = {
    "arc": partial(load_arc, split="test"),
    "winograd": partial(load_winogrande, split="test"),
    "mmlu": partial(load_mmlu, split="test"),
    "hellaswag": partial(load_hellaswag, split="test"),
    "blimp": partial(load_blimp, split="test"),
    "langgame-train": partial(load_langgame, split="train"),
    "langgame-val": partial(load_langgame, split="val"),
}


def load_benchmark(benchmark_name):
    """
    Given the benchmark name, build the benchmark
    """
    return EVALS_DICT[benchmark_name]()
