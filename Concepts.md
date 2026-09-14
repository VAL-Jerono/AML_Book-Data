# Applied Machine Learning — Concept Journey
### DSA 8401 · Labs 02 – 07

> *"A model is only as honest as the data that trained it."*

This document traces the full arc of the course — from raw, messy data all the way to a
convolutional neural network reading chest X-rays. Each lab answers a question the previous
one left open. Read it like a story, because that is what machine learning really is.

---

## Lab 02 — Messy Data & Leakage-Safe Features

### The Opening Question

> *You have a CSV of mobile-money transactions. Can you just feed it to a model?*

The answer is almost never *yes* — and this lab explains why.

---

### 1. Auditing Data — Trust Nothing Until You Verify

Before a single line of modelling code is written, we ask the hard questions:

- Are there duplicate rows?
- Are values missing — and *why* are they missing?
- Are the data types and formats even correct?

In the mobile-money dataset we found **3,600 exact duplicate rows** (same `txn_id`,
repeated). These had to be dropped *before* any train/test split. If you leave duplicates in,
the same event can land in both training and test — your evaluation is a lie.

We also found that `amount` was stored as a string (`"1,500/-"`, `"(227/-)"`, `"4,742/- Dr"`)
and `txn_time` mixed three different date formats. Raw data is never clean.

---

### 2. Missingness — Not All Gaps Are Created Equal

> *Why is this value missing? Does the gap itself contain information?*

There are three mechanisms, and they demand different treatments:

| Mechanism | What it means | What we do |
|-----------|--------------|------------|
| **MCAR** — Missing Completely At Random | Random equipment dropout. No pattern. | Safe to impute |
| **MAR** — Missing At Random | Explained by *other observed* columns. `agent_id` missing only in Mwanza/Jinja/Arusha due to a legacy logger. | Impute within groups |
| **MNAR** — Missing Not At Random | The value itself causes the gap. GPS coordinates missing in remote areas *because* of where they are. | Handle carefully — the gap is signal |

Misclassifying MNAR as MCAR and imputing blindly can introduce a subtle, invisible bias.

---

### 3. Feature Engineering — Building Meaning from Transactions

Raw rows tell you *what happened*. Features tell you *who this person is*.

From individual transactions we aggregate to **customer-level (wallet-level) features**:

- **RFM features** — Recency (when did they last transact?), Frequency (how often?),
  Monetary (how much, total?)
- **Ratio features** — send/receive ratio, cash-in/cash-out balance
- **Cyclical features** — encoding hour-of-day and day-of-week as sin/cos pairs so the
  model understands that 23:59 and 00:01 are neighbours, not opposites

> **Key concept:** Feature engineering is where domain knowledge becomes mathematical signal.

---

### 4. Data Leakage — The Silent Killer

> *The model scores 1.00 AUC. Should you celebrate?*

No. You should be suspicious.

**Data leakage** happens when information from the future (or from the outcome itself) slips
into the training features. The model memorises the answer rather than learning to predict it.

Two planted leaks were hidden in the dataset and found via a **correlation screen**
(rank every feature by |ρ(feature, label)|). The top of the list had two fields written
*after* the transaction was reviewed — fields that would not exist at scoring time in
production. Including them inflated ROC-AUC from **~0.63 → ~1.00**.

That gap — **+0.37 AUC** — is entirely fictitious. In production, the model would collapse.

**Two structural safeguards prevent leakage:**

1. **Pipelines** — every imputer and scaler lives *inside* a `sklearn Pipeline`, re-fit
   on training data only per fold. Nothing crosses the train/validation boundary.
2. **GroupKFold on `customer_id`** — every SIM card belonging to a single customer stays
   on the same side of each split. A customer who appears in training never leaks into
   validation.

> **Key concept:** A leaky metric is worth less than an honest one. Structure your code so
> leakage becomes *structurally impossible*, not just unlikely.

---

### Lab 02 — Concepts Checklist

- [ ] MCAR / MAR / MNAR missingness classification
- [ ] Duplicate detection and removal before splitting
- [ ] Parsing malformed strings (amounts, dates)
- [ ] Customer-level feature aggregation (RFM, ratios, cyclical encoding)
- [ ] Correlation screen for feature-label leakage
- [ ] Sklearn `Pipeline` + `ColumnTransformer` as a leakage safeguard
- [ ] `GroupKFold` cross-validation for grouped/longitudinal data
- [ ] ROC-AUC inflation as a leakage symptom

---

*Lab 02 leaves us with a clean, honest feature matrix and a model that actually generalises.
But how do we know if it's good? What does "good" even mean when fraud is rare?
Lab 03 answers that.*

---

## Lab 03 — Evaluation, Imbalance & Business Cost

### The Opening Question

> *Our model says it's 97% accurate. Why does the fraud team hate it?*

Because 97 out of 100 customers are *not* fraudulent. A model that labels everyone
"legit" is 97% accurate and catches **zero** fraud cases. Accuracy is the wrong ruler.

---

### 1. The Confusion Matrix — Four Outcomes, Not One

Every prediction falls into one of four buckets:

|  | Predicted: Fraud | Predicted: Legit |
|--|--|--|
| **Actual: Fraud** | TP — Caught it ✅ | FN — Missed it ❌ |
| **Actual: Legit** | FP — False alarm ⚠️ | TN — Correctly cleared ✅ |

From these four numbers flow every evaluation metric. Before using any metric, ask yourself:
*which of these four outcomes does this metric care about?*

---

### 2. Precision, Recall & F1 — The Right Ruler for Rare Events

> *When the model raises an alarm, how often is it right? And how many real fraudsters
> does it miss?*

- **Precision** = TP / (TP + FP) — "When we flag someone, are we usually correct?"
- **Recall** = TP / (TP + FN) — "Of all real fraud cases, how many did we catch?"
- **F1 Score** = harmonic mean of precision and recall — useful when you want both

In fraud detection, **recall is usually more important** than precision. A missed fraudster
(FN) costs far more than an unnecessary investigation (FP).

---

### 3. ROC-AUC vs PR-AUC — Choosing the Right Curve

> *Why do two metrics that both measure "how good is the model" give such different stories?*

The **ROC curve** plots True Positive Rate vs False Positive Rate across all thresholds.
It looks great even on imbalanced data — because the huge pool of true negatives keeps
the False Positive Rate artificially low.

The **Precision-Recall (PR) curve** ignores true negatives entirely and focuses only on the
minority class. For fraud — where only ~8% of transactions are fraudulent — **PR-AUC is the
honest metric**. The random baseline is approximately the fraud rate itself, not 0.5.

---

### 4. Calibration — Trustworthy Probabilities

> *The model says there is a 20% chance of fraud. Does that mean 20% of such customers
> are actually fraudulent?*

Not necessarily. Models that discriminate well (high AUC) can still produce poorly scaled
probabilities. **Calibration** is the property where a predicted probability of 20% actually
corresponds to ~20% observed fraud rate.

We measure calibration with the **Brier Score** (lower = better) and visualise it with a
calibration plot. Isotonic calibration can correct a skewed model without retraining it.

Why does this matter? Because **the threshold we set in production is based on probability**.
If the probabilities are wrong, the threshold is wrong, and the business decision is wrong.

---

### 5. Class Imbalance — Three Remedies

> *Fraud is rare. How do we stop the model from ignoring it entirely?*

Three approaches, each with a different philosophy:

| Approach | Mechanism | Trade-off |
|----------|-----------|-----------|
| **Threshold Moving** | Keep the model; change the decision boundary | No retraining; simple and reversible |
| **Class Weighting** | Tell the loss function to penalise FNs more | Model changes; data unchanged |
| **SMOTE-NC** | Create synthetic minority observations | Data changes; risk of overfitting if done outside CV |

> ⚠️ **Critical rule:** SMOTE must happen *inside* the pipeline, re-fit per fold during
> cross-validation. Oversampling the full dataset before splitting leaks synthetic copies
> of validation observations into training.

---

### 6. Business Cost — The Real Objective Function

> *The model with the highest PR-AUC isn't always the cheapest model to run. Why?*

Because metrics don't know about money. In the mobile-money problem:

- **Cost of a missed fraudster (FN)** = KES 10,000
- **Cost of an unnecessary investigation (FP)** = KES 800

Total cost = `FN × 10,000 + FP × 800`

A missed fraudster is **12.5× more expensive** than a false alarm. This asymmetry changes
everything. The mathematically optimal threshold is:

**t\* = C_FP / (C_FP + C_FN) = 800 / 10,800 ≈ 0.074**

We flag a customer whenever the model's fraud probability exceeds **7.4%** — not 50%.
The traditional 0.50 threshold would miss most fraud cases and cost significantly more.

> **Key concept:** The threshold is a business decision, not a machine-learning default.
> Model probabilities + class imbalance + business costs + operational capacity = threshold.

---

### Lab 03 — Concepts Checklist

- [ ] Confusion matrix (TP, FP, TN, FN)
- [ ] Precision, Recall, F1 — when each matters
- [ ] ROC-AUC vs PR-AUC — why PR-AUC wins on imbalanced data
- [ ] Calibration — Brier Score, isotonic calibration
- [ ] Class imbalance: threshold moving, class weighting, SMOTE-NC
- [ ] SMOTE inside the pipeline rule
- [ ] Stratified cross-validation to preserve class ratios
- [ ] Business cost function — asymmetric error costs
- [ ] Cost-optimal threshold formula: t* = C_FP / (C_FP + C_FN)

---

*We now have a rigorous evaluation framework. But our model is Logistic Regression — a
straight line through high-dimensional space. What if the fraud patterns are not linear?
Lab 04 introduces trees, ensembles, and the art of tuning.*

---

## Lab 04 — Trees, Ensembles & Hyperparameter Optimisation

### The Opening Question

> *Logistic Regression is interpretable and fast. Why would we ever replace it?*

Because real-world fraud patterns don't live on a straight line. They interact, combine, and
hide in non-linear thresholds — "amount > 5,000 AND hour between midnight and 3am AND
first transaction from this region." A tree captures that. A logistic model cannot.

---

### 1. The Decision Tree — Interpretable but Unstable

A decision tree splits data at each node using a single feature and threshold.
At depth 4, you can read it like a fraud analyst's mental checklist.

But here is its fatal flaw — **bootstrap instability**. Fit the same tree on two slightly
different resamples of the same data and the root split changes completely. A small shake
in the data produces a completely different tree. This high variance is why we can't trust
a single tree in production.

> **Key concept — Bias vs Variance:**
> A single tree has *low bias* (it fits the training data well) but *high variance*
> (it changes dramatically with small data changes). We need both low bias and low variance.

---

### 2. Random Forest — Variance Reduction Through Averaging

> *What if we trained hundreds of trees on different samples and averaged them?*

That is **Bagging** (Bootstrap Aggregating, Breiman 1996). Random Forest adds one more
trick: at each split, it randomly considers only a *subset* of features. This de-correlates
the trees so their errors cancel rather than compound.

The result: individual trees are still unstable, but the **average** is remarkably stable.
Bagging attacks **variance** — it doesn't change what each tree learns on average,
it just reduces how wildly that changes from sample to sample.

**Feature importance** requires care here:
- **Impurity importance** (built into sklearn) is biased toward high-cardinality features
  because more split points = more apparent importance
- **Permutation importance** shuffles one feature at a time and measures the score drop.
  This is what we trust — it measures actual predictive contribution.

---

### 3. LightGBM — Attacking Bias with Gradient Boosting

> *Random Forest reduced variance. What if the average prediction itself is still wrong?*

Boosting takes a different philosophy: fit trees **sequentially**, each one correcting
the residual errors of the last. Instead of building parallel, independent trees and
averaging them, boosting builds a cumulative model where each new tree is laser-focused
on what the current ensemble still gets wrong.

**LightGBM** is the production-grade implementation:
- **Leaf-wise growth** — splits the leaf with the highest loss reduction, not level by level
- **Native missing value handling** — no manual imputation required
- **Early stopping** — set `n_estimators` very high (e.g. 4,000), monitor validation
  PR-AUC, and stop when it stops improving. This removes `n_estimators` from the
  hyperparameter search entirely.

---

### 4. Monotonic Constraints — Governance, Not Hyperparameters

> *The model predicts lower fraud risk for higher amounts. Is that a bug?*

It could be a genuine pattern — or it could be noise that the model overfit to.
Either way, the business cannot ship a model with that behaviour.

**Monotonic constraints** encode business logic directly into the model:
- Higher risk score from a pre-screener → fraud probability must **increase** (+1)
- Higher post-transaction balance → fraud probability must **decrease** (−1)
- Larger transaction amount → fraud probability must **increase** (+1)

These constraints are not tunable hyperparameters — they come from policy and governance.
LightGBM enforces them natively with minimal PR-AUC loss.

> **Key concept:** A model that violates business logic cannot be deployed, regardless
> of its AUC score. Monotonic constraints make the model auditable and defensible.

---

### 5. Hyperparameter Optimisation with Optuna

> *There are six knobs to tune. How do we search 60 combinations without just guessing?*

Grid search is exhaustive but exponentially expensive. Random search is cheap but dumb.
**Optuna** uses **TPE (Tree-structured Parzen Estimator)** — a Bayesian sampler that
builds a probabilistic model of *which hyperparameter regions produce good results*
and proposes the next trial accordingly.

Key design choices:
- **60 trials** (rule of thumb: ≥10× the number of dimensions)
- **Median pruning** — kills unpromising trials early, giving a 3–10× speedup
- **Metric: PR-AUC** — the metric we actually care about
- **Monotonic constraints frozen** — governance is not tunable

---

### 6. The Winner's Curse — Optimism in HPO

> *The best score from 60 trials is 0.87 PR-AUC. Is that what we'll get in production?*

No — and this is fundamental. The **winner's curse** says: the maximum of 60 noisy scores
is *above* the truth by construction. Every score has random noise. The best score in a set
of 60 is almost certainly an upward fluctuation.

The only honest estimate comes from the **out-of-time test set** — data from *after*
the training period, never touched during training or tuning. This is why temporal
splits matter more than random splits in production ML systems.

---

### 7. Stacking — When Does Complexity Pay?

> *If one model is good, why not combine three?*

**Stacking** trains a meta-learner on the out-of-fold predictions of multiple base models.
The meta-learner learns to weight each model's strengths. In the lab, stacking
(Logistic Regression + Random Forest + tuned LightGBM) added marginal PR-AUC gain.

But the real question is economic: does a small AUC gain justify serving **three models** in
production — tripling the latency, memory, and monitoring burden?

**Recommendation from the lab:** Deploy the tuned LightGBM with monotonic constraints.
Simpler, governable, and nearly as good as the stack.

---

### Lab 04 — Concepts Checklist

- [ ] Decision tree — interpretability vs instability
- [ ] Bias-variance trade-off
- [ ] Bagging / Random Forest — variance reduction through averaging
- [ ] Impurity vs permutation feature importance
- [ ] LightGBM — leaf-wise growth, early stopping, native imbalance handling
- [ ] Monotonic constraints — governance as model architecture
- [ ] Optuna TPE — Bayesian hyperparameter search
- [ ] Median pruning — early trial termination for speed
- [ ] Winner's curse — optimism in cross-validated HPO scores
- [ ] Out-of-time test set — the only honest evaluation
- [ ] Stacking — economic vs statistical trade-off

---

*We've now built increasingly powerful supervised models. But what if we don't have labels
at all? What if no one ever told us which customers were fraudulent?
Lab 05 flips the entire paradigm: unsupervised learning.*

---## Lab 05 — Customer Segmentation & Anomaly Detection (Unsupervised Learning)

### The Opening Question

> *What if we had no fraud labels at all? Could we still find suspicious wallets?*

Labs 02–04 used a label (`fraud = 1 / 0`) to train and evaluate every model.
But labels are expensive — they require human reviewers, legal agreements, and time.
In Lab 05 we take the **same mobile-money data** and ask: what structure can we find
*without any labels?*

This is **unsupervised learning** — the model discovers patterns on its own.

---

### 1. Aggregating to Wallet Level — One Row Per Customer

> *Why can't we cluster raw transactions?*

A single customer may have 50 transactions. Clustering those 50 rows separately would
be clustering *events*, not *people*. We want to understand customer *behaviour*.

So we engineer wallet-level features first: transaction counts by type, amount statistics,
time-of-day patterns, send/receive ratios. Now each row is a behavioural fingerprint
of a single customer.

---

### 2. PCA — Compressing Signal, Removing Noise

> *We have 20+ wallet features. Do we need all of them?*

Many features are correlated. Clustering in high-dimensional space is harder — distances
become noisy and less meaningful (the **curse of dimensionality**).

**Principal Component Analysis (PCA)** projects the data onto new axes (principal components)
that capture the most variance. We keep enough components to retain **95% of total variance**.

Two critical rules:
1. **Standardise first** — PCA is sensitive to scale. A feature measured in thousands
   dominates a feature measured in units.
2. **Fit PCA on training data only** — the same leakage principle from Lab 02 applies here.

The first principal component (PC1) typically captures overall transaction *volume*;
PC2 captures amount *scale*. These give us a 2D map of the customer space.

---

### 3. K-Means — Dividing the Map into Segments

> *How many types of customers are there? And how do we decide?*

K-Means assigns each wallet to one of *k* centroids by minimising within-cluster distance.
But *k* is our choice — the algorithm doesn't know what makes a good number of clusters.

Two diagnostics help:
- **Elbow method** — plot *inertia* (total within-cluster squared distance) vs k.
  Look for the point where adding more clusters stops reducing inertia sharply.
- **Silhouette score** — for each point, compares how similar it is to its own cluster
  vs the nearest other cluster. Higher = better-separated clusters. Range: −1 to +1.

Typical segments found in mobile-money data: low-activity wallets, moderate-use wallets,
high-frequency small-amount wallets, and high-value occasional wallets.

---

### 4. DBSCAN — Density-Based Clustering Without Fixing k

> *What if the clusters don't have spherical shapes? What if we don't know k in advance?*

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** groups points
that are closely packed together and labels sparse outliers as **noise (−1)**.

It requires two parameters:
- **ε (epsilon)** — maximum distance between two neighbours. Chosen from the *k-distance
  plot* — the knee in the sorted distances to the kth nearest neighbour.
- **min_samples** — minimum neighbours to form a core point.

The power of DBSCAN: it finds clusters of arbitrary shape and explicitly identifies
outliers — points that don't belong to any cluster. Those noise points are natural
anomaly candidates.

---

### 5. Gaussian Mixture Models — Probabilistic Anomaly Detection

> *Can we assign a probability score to how "normal" each wallet is?*

**GMMs** model the data as a mixture of Gaussian distributions — one per cluster.
Once fitted, every wallet receives a **log-likelihood score** under the model.

Wallets with very low log-likelihood are anomalous — they fit poorly under any of the
learned Gaussian components. The bottom 0.5% by log-likelihood are flagged as anomalies.

We select the number of components (clusters) using **BIC (Bayesian Information Criterion)** —
which penalises model complexity, preventing us from overfitting the mixture.

The key payoff: GMM anomalies overlap significantly with wallets that have high
human-assigned risk scores — confirming the unsupervised approach catches real outliers.

> **Key concept:** Unsupervised anomaly detection works even without fraud labels.
> It finds *structure deviations*, not just pattern matches.

---

### Lab 05 — Concepts Checklist

- [ ] Wallet-level feature aggregation — from events to behaviour
- [ ] Curse of dimensionality — why high-dimensional clustering is hard
- [ ] PCA — variance retention, standardisation requirement, leakage rule
- [ ] Principal components — what PC1 and PC2 represent
- [ ] K-Means — algorithm, inertia, elbow method, silhouette score
- [ ] DBSCAN — density-based clustering, ε selection, noise points as anomalies
- [ ] k-distance plot — choosing ε
- [ ] Gaussian Mixture Models — log-likelihood scoring, BIC model selection
- [ ] Anomaly detection without labels

---

*Unsupervised learning taught us that structure exists even without labels. Now we push
further — what if the patterns we need to detect are too complex even for hand-crafted
features? Lab 06 introduces neural networks, where the model learns its own features.*

---

## Lab 06 — Neural Networks: MLP for Fraud Detection

### The Opening Question

> *Can a simple neural network compete with a well-tuned tree ensemble on tabular fraud data?*

Trees dominated tabular data for years. But neural networks have one property that trees
lack: they are **differentiable end-to-end**. Every parameter adjusts simultaneously based
on the gradient of the loss. They don't need hand-crafted feature interactions —
they learn them. Lab 06 asks: is that enough to beat LightGBM on our fraud problem?

---

### 1. Why Neural Networks Need Scaled Features

> *Why does a 0-1 scaled input train faster than an unscaled one?*

Neural networks compute weighted sums at each neuron. If one input has values in the
thousands and another in the range 0–1, the gradient updates for those weights will be
wildly different in magnitude. The network spends most of its learning budget adjusting
the large-magnitude weight, starving the small-magnitude one.

**StandardScaler** removes this imbalance by centring each feature at zero mean and
unit variance. The fundamental rule from Lab 02 applies without exception here:
**fit the scaler on training data only — never on validation or test**.

---

### 2. Temporal Splits — The Model Must Learn from the Past

> *Why don't we just do a random 80/20 split?*

In fraud detection, the model will be deployed on *future* transactions.
A random split lets the model see future data patterns during training — another
form of leakage. A **temporal split** respects time:

- Training set: the past
- Validation set: the near future (used for early stopping and threshold selection)
- Test set: the far future (never touched until final evaluation)

This is not a style preference. It is the only evaluation that reflects reality.

---

### 3. MLP Architecture — How Information Flows

> *What does "18 inputs → 32 ReLU → 16 ReLU → 1 Sigmoid" actually mean?*

An **MLP (Multi-Layer Perceptron)** passes data through layers of neurons:

```
18 inputs → [Dense 32, ReLU] → [Dense 16, ReLU] → [Dense 1, Sigmoid]
```

- **Dense layer** — every input connects to every neuron. Parameters = (inputs + 1) × units
- **ReLU (Rectified Linear Unit)** — activation for hidden layers: f(x) = max(0, x).
  Simple, fast, avoids the vanishing gradient problem
- **Sigmoid** — output activation: squashes the final value to [0, 1], giving a
  fraud probability

Parameter count matters:
- Layer 1: (18 + 1) × 32 = **608 parameters**
- Layer 2: (32 + 1) × 16 = **528 parameters**
- Output: (16 + 1) × 1 = **17 parameters**
- **Total: 1,153 learnable parameters**

This is a *tiny* network — deliberately so. On a small tabular dataset, over-parameterised
networks overfit aggressively.

---

### 4. Training — Loss Functions, Optimisers & Class Weights

> *What does the network actually minimise during training?*

- **Binary cross-entropy** — the standard loss for binary classification. It penalises
  confident wrong predictions much more than uncertain ones.
- **Adam optimiser** — an adaptive learning rate optimiser that maintains per-parameter
  momentum. It converges faster than vanilla gradient descent.
- **Class weights** — since fraud is rare (~8%), we pass `class_weight` to the training
  loop so the loss function penalises missed fraud cases more heavily. Same principle
  as Lab 03's class weighting, just implemented differently.

---

### 5. Early Stopping — Knowing When to Stop

> *More training epochs always means a better model, right?*

Wrong. After enough epochs, the model begins to **overfit** — it memorises the training
data and loses the ability to generalise. Validation loss stops decreasing and starts rising.

**Early stopping** monitors the validation metric (we use PR-AUC, not accuracy) and halts
training when it stops improving. This:
- Prevents overfitting
- Removes the number of epochs as a hyperparameter to tune manually
- Saves compute time

The saved best model weights (from the epoch with the best validation PR-AUC) are restored.

---

### 6. Evaluation — PR-AUC as the Primary Metric (Again)

We use **PR-AUC** — the area under the Precision-Recall curve — for exactly the same
reason as in Lab 03. Fraud is rare. Accuracy is misleading. The random baseline for
PR-AUC is approximately the fraud rate.

After choosing a decision threshold on the **validation set**, we evaluate the final
model on the **out-of-time test set** — data the model never saw.

> **Key insight from the lab:** A simple MLP can be competitive with tree ensembles on
> small tabular data — but it requires careful scaling, temporal splitting, and class
> weighting. Trees typically win on larger tabular datasets. Neural networks dominate
> when the inputs are images, text, or audio — as Lab 07 will demonstrate.

---

### Lab 06 — Concepts Checklist

- [ ] Feature scaling for neural networks — why and how
- [ ] Temporal train/validation/test split
- [ ] MLP architecture — Dense layers, ReLU, Sigmoid
- [ ] Parameter counting — (inputs + 1) × units
- [ ] Binary cross-entropy loss
- [ ] Adam optimiser
- [ ] Class weights in neural network training
- [ ] Early stopping — monitoring validation PR-AUC
- [ ] Overfitting — training curve vs validation curve
- [ ] PR-AUC as evaluation metric for imbalanced neural network

---

*The MLP learned from numbers in a table. But what happens when the input is not a table
at all — it's a photograph? A completely different architecture is needed, one that
understands spatial structure. Lab 07: Convolutional Neural Networks.*

---

## Lab 07 — Convolutional Neural Networks: Chest X-Ray Classification

### The Opening Question

> *Can a small CNN tell a NORMAL chest X-ray apart from a PNEUMONIA/COVID-19 one?*

An MLP sees each pixel as an independent number — completely blind to where the pixel lives.
A chest X-ray is 150×150 pixels. That's 22,500 features, almost all of them spatially
correlated with their neighbours. Treating each independently wastes everything the image
tells us about *structure*.

A **CNN** is an architecture designed to exploit spatial structure by learning local
patterns and composing them into global ones.

---

### 1. Why Not an MLP for Images?

> *An MLP has worked on everything so far. Why does it fail on images?*

Two problems:
1. **Scale** — a 224×224 image has 150,000+ pixels. Fully connected layers would need
   billions of parameters just for the first layer.
2. **Translation invariance** — a cat in the top-left corner is still a cat in the
   bottom-right. An MLP has to learn the cat separately for every possible position.
   A CNN shares the same filters across the entire image.

---

### 2. Conv2D — Learning Visual Patterns

> *How does a network learn to "see" edges, textures, and shapes?*

A **Conv2D layer** slides a small filter (e.g. 3×3) across the image and computes a
dot product at every position. Each filter detects one type of local pattern:
an edge, a colour gradient, a texture.

Key properties:
- **Parameter sharing** — the same filter is applied across the entire image.
  A 3×3 filter has only 9 weights regardless of image size.
- **Local connectivity** — each output neuron sees only a small region of the input,
  not the entire image.
- **Stacking layers** — layer 1 detects edges; layer 2 detects shapes made of edges;
  layer 3 detects organs made of shapes. Depth builds abstraction.

---

### 3. MaxPooling — Shrinking Without Losing Structure

> *The feature map after Conv2D is still large. How do we compress it?*

**MaxPooling2D** (typically 2×2) divides the feature map into small windows and keeps
only the maximum value in each. This:
- Halves the spatial dimensions (reducing computation)
- Keeps the strongest signal from each region
- Makes the representation slightly translation-invariant

The CNN architecture in the lab stacks three Conv2D + MaxPooling blocks before
flattening to a vector.

---

### 4. Dropout — Regularisation for Neural Networks

> *The network overfits on only ~150 training images. How do we stop it?*

**Dropout** randomly sets a fraction of neurons to zero during each training step.
The network cannot rely on any individual neuron — it must distribute knowledge
across many. This acts as an implicit ensemble of many different network architectures.

In the lab, Dropout(0.5) was applied before the final classification layer.
Combined with early stopping (monitoring validation accuracy), it significantly
reduced overfitting on the small dataset.

---

### 5. The Full Pipeline: From Folder to Prediction

```
train/NORMAL/*.jpeg          →  image loading (Keras ImageDataset)
train/PNEUMONIA/*.jpeg       →  rescaling 0-255 → 0-1
                             →  [Conv2D → MaxPool] × 3
                             →  Flatten → Dense → Dropout
                             →  Sigmoid → probability of PNEUMONIA
```

**Data loading from folders** — Keras can build a dataset directly from folder structure:
```
train/NORMAL/...
train/PNEUMONIA/...
```
Labels are inferred from the folder names automatically.

**`cache()` and `prefetch()`** — after loading, `cache()` keeps images in memory
so they don't reload from disk each epoch. `prefetch()` prepares the next batch
while the GPU is processing the current one. These improve speed without changing
the model.

---

### 6. Evaluation in a Medical Context

> *The model is 90% accurate. Is that good enough to read X-rays?*

In medical imaging, the confusion matrix interpretation changes:

| Error Type | Clinical Consequence |
|---|---|
| FN — predicted NORMAL, actually PNEUMONIA | Patient goes untreated — potentially fatal |
| FP — predicted PNEUMONIA, actually NORMAL | Unnecessary treatment — harmful but manageable |

A **confusion matrix** shows not just accuracy but *which class* is being confused with
the other. In this context, recall (catching actual PNEUMONIA cases) is paramount.
A model that misses PNEUMONIA to look more accurate is dangerous.

> **Key concept:** The domain shapes the metric. In fraud detection, FNs cost money.
> In radiology, FNs cost lives.

---

### Lab 07 — Concepts Checklist

- [ ] Why MLP fails on image data — scale and translation invariance
- [ ] Conv2D — filter, parameter sharing, local connectivity
- [ ] Feature map — spatial representation after convolution
- [ ] MaxPooling2D — spatial compression, signal retention
- [ ] Stacking Conv+Pool blocks — building abstraction through depth
- [ ] Flatten — feature map to vector
- [ ] Dropout — regularisation through random neuron masking
- [ ] Image rescaling — 0-255 → 0-1
- [ ] Loading images from folder structure with Keras
- [ ] `cache()` and `prefetch()` — pipeline speed optimisation
- [ ] Training curves — detecting overfitting visually
- [ ] Confusion matrix in medical imaging — FN vs FP stakes

---

---

## The Full Story — How Each Lab Feeds the Next

| Lab | Core Question | What It Left Open |
|-----|---------------|-------------------|
| **02** | Can I trust this data? | Can I trust my evaluation? |
| **03** | How do I measure model quality honestly? | Can a straight line capture complex fraud patterns? |
| **04** | Can trees and ensembles beat logistic regression? | What if I don't have labels at all? |
| **05** | Can I find patterns without labels? | Can I learn features automatically instead of engineering them? |
| **06** | Can a neural network learn its own features from numbers? | What if the input is an image, not a table? |
| **07** | Can a CNN learn to see medical patterns? | *(The frontier — transfer learning, larger datasets, deployment)* |

---

### The Three Principles That Never Changed

Across every lab — from data auditing to chest X-ray classification — three ideas
repeated themselves under different names:

1. **Leakage is the enemy.** Whether it is a future-dated column in Lab 02, SMOTE
   applied before CV in Lab 03, or a scaler fit on the test set in Lab 06 — the
   principle is the same. Information that would not be available at deployment time
   must never touch training.

2. **The metric must match the business problem.** Accuracy is usually wrong. PR-AUC
   is usually better. Business cost is often best. And in radiology, even PR-AUC
   is not enough — you need to weight FNs differently.

3. **Complexity must earn its cost.** A tuned LightGBM beat a stacked ensemble on
   economics. A small MLP competed with trees on a small dataset. A simple 3-block
   CNN classified X-rays without ImageNet pretraining. Start simple. Add complexity
   only when the data justifies it.

> *"All models are wrong. Some are useful. The skill is knowing which, when, and why."*
> — George Box (paraphrased)



---

## What was built

**6 narrative sections, each self-contained but deliberately feeding into the next:**

| Section | Central Question | Key Concepts |
|---|---|---|
| **Lab 02** | Can I trust this data? | MCAR/MAR/MNAR, feature engineering, pipeline leakage, GroupKFold |
| **Lab 03** | What does "good" mean for a rare class? | Confusion matrix, PR-AUC, calibration, SMOTE, business cost threshold |
| **Lab 04** | Can trees do what logistic regression can't? | Bias-variance, bagging, LightGBM, monotonic constraints, Optuna, winner's curse |
| **Lab 05** | What if there are no labels? | PCA, K-Means, DBSCAN, GMMs, BIC, log-likelihood anomaly scoring |
| **Lab 06** | Can a neural network learn its own features? | MLP architecture, ReLU/Sigmoid, Adam, early stopping, temporal splits |
| **Lab 07** | What if the input is an image? | Conv2D, MaxPooling, parameter sharing, Dropout, medical FN vs FP stakes |

Each section ends with a **bridge sentence** that plants the question the next lab answers, and closes with a **checklist** of all key concepts per notebook. The document closes with a summary table of the full arc + three universal principles that thread through every lab.