"""Generates CSV to use for results calculation"""

import pandas as pd

turk_res = pd.read_csv("approvals.csv")
articles = pd.read_csv("articles100.csv")

systems = ["bottom-up", "reference", "dca", "pg", "novel", "abs-rl-rerank"]
n_sys = len(systems)

ans_systm = [f"Answer.system_{idx}"       for idx in range(n_sys)]
ans_cover = [f"Answer.Coverage_{idx}"     for idx in range(n_sys)]
ans_merge = [f"Answer.Merging_{idx}"      for idx in range(n_sys)]
ans_faith = [f"Answer.Faithfulness_{idx}" for idx in range(n_sys)]
ans_gramm = [f"Answer.Grammatical_{idx}"  for idx in range(n_sys)]
other_col = ["WorkerId", "Answer.article_hash", "Approve"]
n_systems = [f"n_{sys}" for sys in systems]

results = turk_res[other_col + ans_systm + ans_cover + ans_merge + ans_faith + ans_gramm]
results.set_index(["Answer.article_hash", "WorkerId"], inplace=True)

for bin_col in (ans_cover + ans_faith + ans_gramm):
    results[bin_col] = results[bin_col].map({"YES": 1, "NO": 0})

for system in systems:
    results[f"n_{system}"] = (results[ans_systm] == system).sum(axis=1)

assert (results[[f"n_{sys}" for sys in systems]].sum(axis=0).sum() // n_sys) == results.shape[0]

def n_summaries_per_sys():
    print(results[[f"n_{sys}" for sys in systems]].sum(axis=0))

merge_d = {
    "Bal. Concat.": "bc",
    "Imbal. Concat.": "ic",
    "Replacement": "re",
    "Other": "ot",
}

cols_to_keep = ["Approve"] + n_systems
for sys in systems:
    for col in [f"gramm_{sys}", f"faith_{sys}", f"cover_{sys}"]:
        results[col] = 0
        cols_to_keep.append(col)

    for opt in merge_d.values():
        col = f"merge_{sys}_{opt}"
        results[col] = 0
        cols_to_keep.append(col)


for row, vals in results.iterrows():
    for idx in range(len(systems)):
        sys = vals[f"Answer.system_{idx}"]
        results.loc[row, f"gramm_{sys}"] += vals[f"Answer.Grammatical_{idx}"]
        results.loc[row, f"faith_{sys}"] += vals[f"Answer.Faithfulness_{idx}"]
        results.loc[row, f"cover_{sys}"] += vals[f"Answer.Coverage_{idx}"]
        opt = vals[f"Answer.Merging_{idx}"]
        results.loc[row, f"merge_{sys}_{merge_d[opt]}"] += 1

results = results[cols_to_keep]

results.to_csv("processed.csv")

def faith_stats():
    df = pd.read_csv("processed.csv")
    df.set_index(["Answer.article_hash", "WorkerId"], inplace=True)

    faith = [f"faith_{sys}" for sys in systems]
    sums = df[faith].sum(axis=0)
    print(sums)
