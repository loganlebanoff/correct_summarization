"""Gathers all statistics for paper and makes plots."""
import itertools

import pandas as pd

processed = pd.read_csv("processed.csv")
processed.set_index(["Answer.article_hash", "WorkerId"], inplace=True)

systems = ["bottom-up", "reference", "dca", "pg", "novel", "abs-rl-rerank"]
n_sys = len(systems)

merge_d = {
    "Bal. Concat.": "bc",
    "Imbal. Concat.": "ic",
    "Replacement": "re",
    "Other": "ot",
}

n_systems = [f"n_{sys}" for sys in systems]
sys_gramm = [f"gramm_{sys}" for sys in systems]
sys_faith = [f"faith_{sys}" for sys in systems]
sys_cover = [f"cover_{sys}" for sys in systems]
sys_merge = [f"merge_{sys}_{opt}" for sys, opt in itertools.product(systems, merge_d.values())]
