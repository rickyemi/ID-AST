# Model card: multi-tier Gram / Growth / MIC cascades

| Item | Detail |
|---|---|
| Cascades | `multitier_xgboost.joblib`, `multitier_svm_rbf.joblib`, `multitier_transformer.joblib`: each holds a Tier 1 and a Tier 2 pipeline, their thresholds, feature lists and the breakpoint table |
| Tier models | `tier1_gram_<model>.joblib`, `tier2_growth_<model>.joblib`: full scikit-learn Pipelines (preprocessing + classifier) |
| Preprocessors | `tier1_gram_preprocessor.joblib`, `tier2_growth_preprocessor.joblib` (standalone, for inspection) |
| Metadata | `model_metadata.json`: features, tuned hyperparameters, out-of-fold AUC, thresholds, library versions |
| Input | Raw well rows (IDs, Antimicrobial, Well_DSI and the measured features); missing values allowed |
| Output | Tier 1: P(Gram Positive) per well, pooled per isolate. Tier 2: P(Growth) per well. Tier 3: MIC, DSI and S/I/R per Gram-positive isolate × drug |
| Training data | 5,780 wells from 219 isolates (80% isolate-grouped split); 1,540 wells from 55 isolates held out |
| Versions | Python 3.11, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.14.0 (use the pinned `requirements.txt`) |

## Loading
```python
import joblib, pandas as pd
cascade = joblib.load("models/multitier_svm_rbf.joblib")     # needs `pip install -e .` for the custom classes
wells = pd.read_csv("new_wells.csv", keep_default_na=False, na_values=[""])
out = cascade.predict(wells)
out["gram"]   # per isolate: P_Gram_Positive, Gram_Call
out["wells"]  # per well:    P_Growth, Growth_Call
out["mic"]    # per GP isolate x drug: pred_dsi, pred_mic, pred_cat (S/I/R)
```

## How Tier 3 works
The MIC call is the dilution cut-point that maximises the likelihood of the Tier 2 growth probabilities across the whole series (wells below the cut grow, wells from it up do not). A single artefact well far from the MIC therefore cannot move the call. Isolates that Tier 1 calls Gram Negative get no MIC.

## Intended use and limitations
* Research prototype and decision support for laboratory review. It is **not** a cleared diagnostic, and it must not replace reference broth microdilution or CLSI/EUCAST interpretation by qualified staff.
* Trained and tested on **synthetic** data. The results show that the pipeline and its evaluation work; real-world validation against reference BMD on clinical isolates is required.
* **VME uncertainty:** 0 VMEs among about 40 resistant Test series still leaves a 95% upper bound near 9%. A validation study needs far more resistant isolates.
* **Known weak points (stress tests):** MIC calls degrade when growth-imaging artefacts rise above the training rate, and exact agreement drops with extra missing data. Use image-quality flags and route conflicting series for re-reading.
* Several drug × organism breakpoints are surrogate study cut-offs (no current CLSI MIC breakpoint, or intrinsic resistance); see `data/external/AST_breakpoints_reference.csv`.
