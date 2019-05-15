"""Used to approve/reject MTurk submissions"""

import pandas as pd

results = pd.read_csv("results.csv")

accept = "x"
reject = "Incorrect response to validation questions."

lookup_cols = [f"Answer.system_{idx}" for idx in range(6)]
revers_cols = [(f"Answer.Faithful_{idx}", f"Answer.Grammatical_{idx}")
               for idx in range(len(lookup_cols))]

# find the reference summary idx
lookup_mask = results[lookup_cols] == "reference"
lookup_maxs = lookup_mask.idxmax(axis=1).str.replace("Answer.system_", "").astype(int)

# Determine if they passed the reference "faithfulness" test
faith_rename = [c.replace("system", "Faithfulness") for c in lookup_cols]
revers_mask_faith = results[faith_rename].apply(lambda x: x.map({"YES": 1, "NO": 0})).astype(bool)
revers_mask_faith = revers_mask_faith.values[lookup_mask]

# Determine if they passed the reference "grammaticality" test
gramm_rename = [c.replace("system", "Grammatical") for c in lookup_cols]
revers_mask_gramm = results[gramm_rename].apply(lambda x: x.map({"YES": 1, "NO": 0})).astype(bool)
revers_mask_gramm = revers_mask_gramm.values[lookup_mask]

overall_pass = (revers_mask_faith & revers_mask_gramm)

results["Approve"] = overall_pass
results["Reject" ] = ~overall_pass

results["Approve"] = results["Approve"].map({True: accept, False: ""})
results["Reject" ] = results["Reject" ].map({True: reject, False: ""})

for group, v in results.groupby("WorkerId"):
    print(v["Approve"].value_counts(normalize=True)[0])
    if v["Approve"].value_counts(normalize=True)[0] > 0.7:
        results.loc[results["WorkerId"] == group, "Approve"] = "x"
        results.loc[results["WorkerId"] == group, "Reject" ] = ""
    print(results.loc[results["WorkerId"] == group, "Approve"].value_counts(normalize=True)[0])
    print("---")

results.to_csv("approvals.csv")
results.loc[results["Approve"] == "x"].to_csv("processed.csv")
