# Multi-tier Machine Learning for Digital-Microscopy ID/AST: Gram Reaction, Growth and MIC Calls

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20SVM--RBF%20%7C%20FT--Transformer-orange)

**Author:** YO ([@rickyemi](https://github.com/rickyemi)) · **Status:** research prototype on synthetic data

## Goal
Rapid digital-microscopy systems image each broth-microdilution well within hours. This project builds a reproducible, production-ready **three-tier cascade** that turns those images into lab results. **Tier 1** calls the Gram reaction, **Tier 2** calls Growth / No Growth in every well, and **Tier 3** calls the MIC (with its Dilution Series Index and S/I/R category) for *S. aureus*, *S. epidermidis*, *E. faecalis* and *E. faecium* against vancomycin, ceftriaxone, ampicillin, oxacillin, daptomycin and meropenem. Three classifier families (XGBoost, RBF-SVM, FT-Transformer) are compared.

## Data
| Item | Value |
|---|---|
| Source | Synthetic (`src/data/generate_synthetic_data.py`, seed 42); no real patients or isolates |
| Unit | One well = isolate × drug × two-fold concentration; 7,320 wells × 30 columns, 274 isolates, **8% missing** at random |
| Features | Gram-stain imaging (crystal violet, safranin, hue, wall thickness), ID tests (KOH, morphology, catalase), time-lapse growth imaging (cell counts, fold change, area, elongation, time to detection, OD600), panel design, QC |
| Targets | **T1** Gram Positive 55% / Negative 45% · **T2** Growth 54.6% / No Growth 45.4% · **T3** MIC (17 classes incl. off-scale) + DSI for 466 Gram-positive isolate-drug series |
| Split | 80/20 **by isolate**, stratified by organism: Train 219 isolates (5,780 wells, 367 MIC series) / Test 55 isolates (1,540 wells, 99 series) |

## Methodology
| Stage | What was done |
|---|---|
| **Data analysis** | Gram features: Mann-Whitney U + BH-FDR, effect sizes, single-feature AUC; ID tests: chi-square / Cramér's V; MIC distributions vs breakpoints, antibiogram, dose-response around the MIC, genotype-phenotype concordance |
| **Preprocessing** (Train only) | log10 of counts → **outliers** capped at Tukey IQR fences → **missing values**: median / most frequent (MCAR check passed; mean, median, KNN and iterative imputers tie on grouped-CV AUC) → **z-score** → one-hot |
| **Feature selection** (per tier) | FDR q < 0.05 **and** mutual information ≥ 0.01, then drop near-duplicates (abs Spearman rho > 0.95). Tier 1 keeps 9 of 10 (Organism and Resistance_Marker excluded as leaky); Tier 2 keeps 9 of 15 |
| **Model development** | Tiers 1 and 2: **XGBoost** (RandomizedSearchCV, 30 candidates) and **RBF-SVM** (20 candidates, Platt probabilities), both with 5-fold **GroupKFold by isolate** and a train-validation AUC gap guard ≤ 0.03; **FT-Transformer** (feature tokens, 2 layers × 4 heads, early stopping). Threshold: Youden's J on out-of-fold predictions. **Tier 3**: MIC = maximum-likelihood cut-point of the Tier 2 growth probabilities over the whole series; Tier 1 pools well images per isolate and gates which isolates get an MIC |
| **Performance evaluation** | Train vs Test; ROC, confusion, calibration, MIC agreement, per organism × drug agreement, XGBoost gain and permutation importance; 12 stress tests with pass rules fixed before running |
| **Acceptance criteria** (Test) | Accuracy, sensitivity, specificity, PPV, NPV, AUC, F1 ≥ 0.92 (Tiers 1, 2); EA, AA, CA ≥ 0.92 and **VME < 2%**, ME ≤ 3% (Tier 3; FDA guidance for context: EA and CA ≥ 90%, VME ≤ 1.5%) |
| **Primary endpoints** | (1) Tier 1 Gram accuracy; (2) Tier 2 growth accuracy; (3) Tier 3 EA and VME, end-to-end (an isolate wrongly called Gram Negative gets no MIC and counts as a failure) |
| **Secondary endpoints** | Sensitivity, specificity, PPV, NPV, AUC, F1; isolate-level Gram accuracy; AA, CA, ME, mE; agreement by organism and drug; calibration; stress-test results |

## Results
**Train vs Test** (positive class: Gram Positive / Growth)

| Tier | Metric | XGBoost Train | XGBoost Test | SVM-RBF Train | SVM-RBF Test | Transformer Train | Transformer Test |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 Gram | Accuracy | 0.980 | 0.973 | 0.972 | 0.968 | 0.968 | 0.968 |
| 1 Gram | Sensitivity | 0.967 | 0.966 | 0.971 | 0.971 | 0.952 | 0.959 |
| 1 Gram | Specificity | 0.995 | 0.981 | 0.973 | 0.963 | 0.987 | 0.980 |
| 1 Gram | PPV / NPV | 0.995 / 0.961 | 0.985 / 0.959 | 0.977 / 0.965 | 0.971 / 0.963 | 0.989 / 0.945 | 0.983 / 0.950 |
| 1 Gram | AUC / F1 | 0.999 / 0.981 | 0.996 / 0.975 | 0.997 / 0.974 | 0.994 / 0.971 | 0.997 / 0.970 | 0.994 / 0.971 |
| 2 Growth | Accuracy | 0.994 | 0.985 | 0.989 | 0.983 | 0.990 | 0.979 |
| 2 Growth | Sensitivity / Specificity | 0.996 / 0.992 | 0.989 / 0.980 | 0.990 / 0.987 | 0.987 / 0.979 | 0.992 / 0.987 | 0.983 / 0.975 |
| 2 Growth | PPV / NPV | 0.993 / 0.995 | 0.983 / 0.987 | 0.990 / 0.987 | 0.982 / 0.985 | 0.989 / 0.991 | 0.978 / 0.980 |
| 2 Growth | AUC / F1 | 1.000 / 0.994 | 0.999 / 0.986 | 0.999 / 0.990 | 0.998 / 0.984 | 0.999 / 0.991 | 0.999 / 0.981 |
| 3 MIC | EA | 0.997 | **1.000** | 0.995 | **1.000** | 0.992 | **1.000** |
| 3 MIC | AA | 0.975 | 0.949 | 0.948 | **0.970** | 0.948 | 0.899 ✗ |
| 3 MIC | CA | 0.989 | 0.980 | 0.986 | **1.000** | 0.986 | 0.980 |
| 3 MIC | VME / ME / mE | 0 / 0.5% / 0.8% | **0 / 0 / 2.0%** | 0 / 1.0% / 0.8% | **0 / 0 / 0** | 0 / 1.0% / 0.8% | **0 / 0 / 2.0%** |

* **56 of 57 acceptance criteria met on Test.** All three primary endpoints pass for every model. The one miss is **Transformer AA 0.899** (its EA, CA and VME pass).
* **Recommended cascade: SVM-RBF.** It is the only model with 100% EA and 100% CA on Test and the highest AA (0.970); XGBoost is best on Tiers 1 and 2. Train and Test agree within about 0.015 for Tiers 1 and 2, so there is no overfitting.
* Pooling well images per isolate gives **100% isolate-level Gram accuracy**, so no Gram-positive series lost its MIC in the cascade.
* **Top features** (XGBoost gain and permutation importance): Tier 1: KOH string test, cell elongation, cell area, catalase. Tier 2: OD600, growth fold change, time to detection, occupied area.
* Exact agreement varies by subgroup with small n (e.g. Transformer meropenem AA 0.77, n = 17); EA is 1.00 in every organism and drug subgroup.

**Stress tests (12 tests, 13 rules):** XGBoost 12/13, Transformer 12/13, SVM-RBF 11/13 passed.

| Passed by all 3 models | Not passed |
|---|---|
| Accuracy CI lower bound ≥ 0.92; EA CI lower bound ≥ 0.92; +10% failed Gram reads (isolate accuracy 1.00); 0.25 SD noise; inoculum ± 0.5 log; worst organism / drug subgroup; 10% label noise; permutation test (AUC ≈ 0.99 vs ≈ 0.50 null); Train-Test gap; grouped-CV stability; edge cases | **+5% extra growth-imaging artefacts:** EA 0.88 / 0.89 and VME 5% (2 of 40) for XGBoost / SVM-RBF. **+20% extra missing cells:** AA drop 0.11 (SVM-RBF) and 0.06 (Transformer) |

> **Caveats.** (1) With only about 40 resistant Test series, 0 VMEs still leaves an exact 95% upper bound near 9%. Showing VME < 2% with confidence needs about 180+ resistant isolates. (2) MIC calls are sensitive to image artefacts next to the MIC well, so production use needs image-quality flags and re-reads of conflicting series. (3) The data are synthetic and several breakpoints are surrogate cut-offs. These results show that the pipeline works; clinical validity still has to be tested against reference BMD on real isolates.

---

### Run it
```bash
make install        # pinned dependencies (CPU only)
make all            # data → analysis → features → train → evaluate → stress → predict (~15 min, 2 CPU cores)
make test           # 17 unit / integration tests
make notebooks      # execute the 5 notebooks in place
python -m src.models.predict_model --input new_wells.csv --model SVM-RBF
docker build -t gram-id-ast-ml . && docker run --rm -v "$PWD":/app gram-id-ast-ml make all
```

### Repository layout
```
├── LICENSE · Makefile · README.md · requirements.txt · pyproject.toml
├── Dockerfile · docker-compose.yml · .github/workflows/ci.yml
├── data/        raw/ (synthetic CSV) · external/ (breakpoints) · interim/ · processed/
├── docs/        IDAST_MultiTier_OnePager.docx
├── models/      tier1_gram_*.joblib · tier2_growth_*.joblib · multitier_*.joblib (3 cascades)
│                tier*_preprocessor.joblib · model_metadata.json · MODEL_CARD.md
├── notebooks/   01 Gram/ID/AST analysis · 02 preprocessing & feature selection · 03 training
│                04 multi-tier evaluation · 05 stress tests
├── references/  data_dictionary.md · methods_references.md
├── reports/     figures/ (31 PNGs) · tables/ (CSV / MD results) · predictions/
├── src/         config.py · pipeline.py · utils.py
│   ├── data/          make_dataset.py · generate_synthetic_data.py
│   ├── analysis/      gram_id_analysis.py · ast_analysis.py · stats.py
│   ├── features/      build_features.py
│   ├── models/        train_model.py · transformer_model.py · multitier.py · ast_metrics.py
│   │                  evaluate_model.py · stress_test.py · predict_model.py
│   └── visualization/ visualize.py
└── tests/       test_data.py · test_ast_metrics.py · test_features.py · test_models.py
```
