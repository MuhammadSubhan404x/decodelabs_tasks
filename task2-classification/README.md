# Heart Disease Classifier

A supervised learning pipeline that predicts heart disease risk from clinical data.
Uses five models (including a majority-class baseline), cross-validation, and SHAP
explainability to go beyond simple accuracy reporting.

## How to run

```bash
cd task2-classification
pip install -r ../requirements.txt

python heart_disease.py
```

Run it and it goes straight through: loads data, runs EDA, compares models, evaluates the
best one, generates all charts, then prompts for a patient risk assessment at the end.

To explore the analysis step-by-step with visualizations rendered inline, open the notebook:

```bash
jupyter notebook analysis.ipynb
```

## Technical highlights

- **DummyClassifier baseline**: included alongside the real models  -  if your model can't beat
  majority-class guessing by a decent margin, something's wrong
- **sklearn Pipeline**: imputation and scaling go inside the pipeline so they don't touch
  the test fold during cross-validation
- **SHAP values**: wanted to see which features were actually driving predictions, not just
  which features the tree split on most often (those can be misleading)
- **Learning curve**: added this to check if getting more patient data would actually help
- **Feature engineering**: tried 3 interaction features, measured the CV impact, they didn't
  help here so the script reports that and uses the original features

## Results

```
Model Comparison (5-Fold Stratified CV)
Model                            Accuracy      Std    vs Baseline
-----------------------------------------------------------------
Baseline (Majority)                0.5413   0.0053               - 
Logistic Regression                0.8316   0.0496         +29.0%
K-Nearest Neighbors                0.8218   0.0262         +28.1%
Random Forest                      0.8447   0.0456         +30.3%  << Best
Support Vector Machine             0.8381   0.0390         +29.7%

Best model AUC: 0.951
```

## Output charts

All charts are pre-generated in `output/` so results are visible without running the script.

| File | Contents |
|------|----------|
| `correlation_heatmap.png` | Feature correlation matrix |
| `confusion_matrix.png` | Prediction errors on test set |
| `roc_curve.png` | ROC curve with AUC |
| `feature_importance.png` | RF importance vs. permutation importance |
| `learning_curve.png` | Training vs. CV score by dataset size |
| `shap_summary.png` | Per-patient SHAP feature contributions |

## Dataset

UCI Cleveland Heart Disease dataset  -  303 patients, 13 clinical features, binary target
(0 = no disease, 1 = disease present). Publicly available from the UCI ML Repository.

## Dependencies

`scikit-learn` | `pandas` | `numpy` | `matplotlib` | `seaborn` | `shap` | `jupyter`
