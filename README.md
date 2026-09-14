# AML YR 26 — Lab 2: Messy Data & Leakage-Safe Features

DSA 8401 Applied Machine Learning (Chapter 2). Worked solution for Lab 2 on a synthetic
mobile-money transaction dump: audit the mess, clean and aggregate it, hunt the leak, and
wrap everything in a leakage-safe scikit-learn pipeline.


---

git fetch upstream

git merge upstream/master

git push origin master



## Contents

| Path | What it is |
|---|---|
| `Data/mobile_money_statements.csv` | Synthetic mobile-money transactions (~183k rows, 18 columns, `is_fraud` target) |
| `Data/mobile_transt.txt` | Data dictionary for the CSV |
| `notebook/lab2_solution.ipynb` | The full worked solution |

## The four tasks

1. **Audit** — missingness per column, exact duplicates, validity violations, and an
   MCAR / MAR / MNAR classification for each incomplete column (`agent_id` = MAR,
   `device_id` = MCAR, `gps_lat`/`gps_lon` = MNAR).
2. **Clean & aggregate** — deduplicate, parse the string `amount` (`"1,500/-"`,
   `"(227/-)"`, `"4,742/- Dr"`) and the three `txn_time` formats, resolve entities to a
   canonical `customer_id` via `reg_id`, then build customer-level RFM, ratio and cyclical
   features as of `SCORING_TS`.
3. **Hunt the leak** — correlation screen plus a lineage argument identifying
   `manual_review_score` and `settlement_status` as post-outcome fields.
4. **Pipeline** — one `ColumnTransformer` + `Pipeline` evaluated with `GroupKFold` grouped
   by `customer_id`, scored with and without the leaky columns.

**Headline result:** ROC–AUC is ~0.63 on the honest feature set and ~1.00 once the leaky
columns are added — roughly +0.37 of pure illusion, since those fields are only written
*after* a transaction is scored.

## Running it

```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook notebook/lab2_solution.ipynb
```

Note: the load cell currently uses an absolute path to `Data/mobile_money_statements.csv` —
change it to a relative path if you clone this elsewhere.
