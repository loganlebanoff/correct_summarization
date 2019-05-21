"""Generates CSV to use for results calculation"""
import hashlib

import pandas as pd
import numpy as np

turk_res = pd.read_csv("approvals.csv")
articles = pd.read_csv("articles100.csv")

systems = ["bottom-up", "reference", "dca", "pg", "novel", "abs-rl-rerank"]
n_sys = len(systems)

inp_artcl = [f"Input.article_content"]
ans_systm = [f"Answer.system_{idx}"       for idx in range(n_sys)]
ans_cover = [f"Answer.Coverage_{idx}"     for idx in range(n_sys)]
ans_merge = [f"Answer.Merging_{idx}"      for idx in range(n_sys)]
ans_faith = [f"Answer.Faithfulness_{idx}" for idx in range(n_sys)]
ans_gramm = [f"Answer.Grammatical_{idx}"  for idx in range(n_sys)]
ans_sumry = [f"Answer.summary_{idx}"      for idx in range(n_sys)]
ans_sents = [F"Answer.sources_{idx}"      for idx in range(n_sys)]
other_col = ["WorkerId", "Answer.article_hash", "Approve"] + inp_artcl + ans_sents
n_systems = [f"n_{sys}" for sys in systems]

system_id = [f"sys_{idx}" for idx in range(n_sys)]
sys_cover = [f"cover_{idx}" for idx in range(n_sys)]
sys_faith = [f"faith_{idx}" for idx in range(n_sys)]
sys_gramm = [f"gramm_{idx}" for idx in range(n_sys)]
sys_merge = [f"merge_{idx}" for idx in range(n_sys)]
sys_sumry = [f"summary_{idx}" for idx in range(n_sys)]
sys_sha256 = [f"sha_{idx}" for idx in range(n_sys)]
sys_sents = [f"src_{idx}" for idx in range(n_sys)]

results = turk_res.loc[:, other_col + ans_systm + ans_cover + ans_merge + ans_faith + ans_gramm + ans_sumry]
results.set_index(["Answer.article_hash", "WorkerId"], inplace=True)
results = results[results["Approve"] == "x"]
results.drop(columns=["Approve"], inplace=True)

for sha in sys_sha256:
    results[sha] = ""

for bin_col in (ans_cover + ans_faith + ans_gramm):
    results.loc[:, bin_col] = results[bin_col].map({"YES": 1, "NO": 0})

for system in systems:
    results.loc[:, f"n_{system}"] = (results[ans_systm] == system).sum(axis=1)

# assert (results[[f"n_{sys}" for sys in systems]].sum(axis=0).sum() // n_sys) == results.shape[0]

merge_d = {
    "Bal. Concat.": "bc",
    "Imbal. Concat.": "ic",
    "Replacement": "re",
    "Other": "ot",
}

cols_to_keep = inp_artcl + n_systems + system_id + sys_sents + sys_cover + sys_faith + sys_gramm + sys_merge + sys_sumry

col_dict = [zip(ans_systm, system_id), zip(ans_cover, sys_cover), 
            zip(ans_merge, sys_merge), zip(ans_faith, sys_faith), 
            zip(ans_gramm, sys_gramm), zip(ans_sumry, sys_sumry),
            zip(ans_sents, sys_sents)]

new_cols = {}
for col_map in col_dict:
    new_cols.update({k: v for k, v in col_map})

results.rename(columns=new_cols, inplace=True)

for merge in sys_merge:
    results.loc[:, merge] = results.loc[:, merge].map(merge_d)

for idx in range(n_sys):
    results[f"sha_{idx}"] = results[[system_id[idx], sys_sumry[idx]]].apply(
            lambda x: hashlib.sha256((x[0] + x[1]).encode()).hexdigest(), axis=1)

results = results[cols_to_keep + sys_sha256]

put_mapping = [(sys_sumry, object), (sys_gramm, int),
               (sys_merge, object), (sys_faith, int), 
               (system_id, object), (sys_cover, int),
               (sys_sents, object)]
for group, rows in results.groupby("Answer.article_hash"):
    parent = rows.iloc[0]
    parent_summ = list(parent[sys_sha256])
    
    for index, child in rows.iloc[1:].iterrows():
        child_summ = list(child[sys_sha256])
        shift = [parent_summ.index(child_summ[idx]) for idx in range(n_sys)]
        
        for mapp, dtype in put_mapping: 
            new = np.arange(n_sys, dtype=dtype) 
            np.put(new, shift, child[mapp].values)
            if mapp == sys_sumry and any(new != parent[sys_sumry]):
                print(child)
            if (shift != np.arange(n_sys, dtype=int)).all():
                results.loc[index, mapp] = new

results.to_csv("processed.csv")
