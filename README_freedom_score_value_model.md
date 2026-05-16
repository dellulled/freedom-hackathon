# Freedom Score — ML-Based User Value Model

## Overview

This Jupyter notebook implements a machine learning-based user value scoring system called the **Freedom Score**. It's designed to predict and segment users by their value and potential, combining user behavior, transactions, and product usage into a single, interpretable score.

The model addresses the SAPP hackathon criterion:
> **ML-based User Value Score (Freedom Score)**  
> Develop an ML model that aggregates user behavior, transactions, and product usage into a single score reflecting user value and potential.

## Key Features

- **LightGBM Classification Model**: Binary classifier predicting high-value users (top 30%)
- **Transparent Value Proxy**: Business-driven aggregation of spending, frequency, engagement, and ecosystem breadth
- **Customer-Level Modeling**: One row per customer to prevent data leakage
- **Robust Validation**: 5-fold cross-validation with customer-level train/test splits
- **Feature Importance**: Gain, permutation, and SHAP-based importance analysis
- **User Segmentation**: Automatic classification into High/Medium/Low value segments
- **Production-Ready Artifacts**: Exportable model bundle and scored customer dataset

## Data Processing

### Input Data
The model expects a CSV file with user-level behavioral data including:
- **Profile fields**: `customer_id`, `customer_age`, `city`, `gender`, `reg_date`, `acq_channel`
- **Transaction data**: `txn_count`, `txn_total_spend`, `txn_avg_spend`, `txn_recency_days`, etc.
- **Process/App usage**: `proc_total_attempts`, `proc_completion_rate`, `proc_unique_processes`, etc.
- **Ecosystem signals**: `is_superapp_adopter`, `pp_purchase_count`, `pp_total_spend`, etc.

### Data Cleaning & Aggregation
1. **Handles duplicated customer records** by aggregating to one row per customer
2. **Imputes missing values**: Zeros for inactivity domains, median for process metrics
3. **Engineers features**:
   - `spend_cv`: Spend volatility
   - `txn_intensity`: Transaction frequency per account age
   - `purchase_dominance`: Share of purchase operations
   - `app_engagement_depth`: Process completion rate × process diversity
   - `ecosystem_breadth`: Number of partner apps + superapp adoption signal
   - `recent_txn_signal`: Recency decay function

## Value Scoring Methodology

### Transparent Value Proxy
A weighted combination of six normalized components:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Spend** | 25% | Log-normalized total transaction spending |
| **Frequency** | 20% | Transaction count indicating engagement |
| **App Depth** | 20% | Process completion rate × process diversity |
| **Product Breadth** | 15% | Diversity of processes used |
| **Ecosystem** | 15% | Partner purchases + superapp breadth |
| **Recency** | 5% | Recent activity signal |

Users in the top 30% by value proxy become the positive class for model training.

## Model Architecture

**LightGBM Classifier** with hyperparameters optimized for binary classification:
- **Estimators**: 600 (with early stopping at 50)
- **Learning Rate**: 0.04
- **Max Depth**: 63 leaves, unlimited depth
- **Min Child Samples**: 80
- **Subsample/Colsample**: 0.85 each
- **Regularization**: L1=0.1, L2=1.0
- **Scale Pos Weight**: Balanced by class distribution

## Model Performance

Typical metrics on hold-out test set (20% of data):
- **AUC-ROC**: ~0.85+ (varies by data quality)
- **Average Precision**: High precision for top-value users
- **5-Fold CV AUC**: Stable across folds (validates generalization)

## Output: Freedom Score

The model output is converted to a **0-100 Freedom Score**:

```
Freedom Score = P(high-value user) × 100
```

### User Segments

| Segment | Score Range | Size | Meaning | Recommended Action |
|---------|-------------|------|---------|-------------------|
| **High** | Top 30% | ~30% | Highest modeled value, strong product usage | Retention, premium offers, loyalty, cross-sell |
| **Medium** | Middle 40% | ~40% | Good potential but undermonetized | Personalized offers, product onboarding |
| **Low** | Bottom 30% | ~30% | Low activity or weak engagement | Activation, education, low-friction nudges |

## Notebook Sections

### 1. **Imports & Configuration**
   - Dependencies: pandas, scikit-learn, LightGBM, SHAP (optional)
   - Sets data path and output directory

### 2. **Load & Validate Data**
   - Loads raw CSV and checks for duplicate customers
   - Analyz duplicates to understand data granularity issues

### 3. **Aggregate to Customer Level**
   - Builds one-row-per-customer modeling table
   - Applies business logic: sum/mean/max/mode aggregations per domain

### 4. **Create Value Proxy**
   - Defines transparent scoring rules for "high-value"
   - Normalizes and weights behavior components

### 5. **Feature Engineering**
   - Selects 40+ modeling features (behavior, engagement, ecosystem signals)
   - Imputes and validates feature matrix

### 6. **Train/Test Split**
   - Customer-level stratified split (80/20)
   - Balances class distribution

### 7. **Train LightGBM Model**
   - Fits model with early stopping on validation AUC
   - Outputs best iteration

### 8. **Evaluate Performance**
   - Computes AUC, average precision, classification metrics
   - Displays confusion matrix

### 9. **Cross-Validation**
   - 5-fold stratified cross-validation
   - Confirms model stability and generalization

### 10. **Feature Importance**
   - Gain importance (split quality contribution)
   - Permutation importance (AUC impact)
   - SHAP values (if library available)

### 11. **Segmentation & Profiles**
   - Creates Freedom Score and value segments
   - Profiles each segment by behavior

### 12. **Export Artifacts**
   - Saves model bundle (model, imputer, metadata)
   - Exports scored customer dataset
   - Exports feature importance

## Output Files

| File | Description |
|------|-------------|
| `freedom_score_value_model.pkl` | Serialized model bundle with imputer and metadata |
| `freedom_scored_users_value.csv` | All customers with Freedom Scores and segment assignments |
| `freedom_score_feature_importance.csv` | Feature importance rankings |

## Usage & Deployment

### Prerequisites
```bash
pip install pandas scikit-learn lightgbm joblib
# Optional: pip install shap
```

### Running the Notebook
1. Ensure CSV data file path is correct in **Section 0**
2. Execute cells sequentially from top to bottom
3. Review evaluation metrics and feature importance
4. Examine segment profiles and business insights

### Loading Trained Model
```python
import joblib

model_bundle = joblib.load("freedom_score_value_model.pkl")
model = model_bundle["model"]
imputer = model_bundle["imputer"]
feature_cols = model_bundle["feature_cols"]

# Score new customers (must have same features)
X_new = impute(X_new, imputer)
scores = model.predict_proba(X_new)[:, 1] * 100
```

## Key Design Decisions

1. **Customer-Level Aggregation**: Prevents row-level duplication from biasing results
2. **No Data Leakage**: Final target and value proxy excluded from model features
3. **Transparent Proxy**: Value is defined by business logic, not discovered by model alone
4. **Balanced Hyperparams**: Conservative regularization to avoid overfitting on small positive class
5. **Stratified Splits**: Preserves class balance in train/test/cross-validation

## Business Insights Enabled

- **Retention**: Identify and protect high-value customers with churn risk
- **Pricing**: Personalize offers and product tiers by segment
- **Onboarding**: Recommend features to Medium segment users based on feature importance
- **Partner Strategy**: Understand ecosystem breadth impact on user value
- **Cohort Analysis**: Track Freedom Score changes over time for behavioral segments

## Troubleshooting

**Issue**: Model package path not found  
**Solution**: Update `DATA_PATH` in section 0 to correct CSV location

**Issue**: SHAP not installed  
**Solution**: Falls back gracefully; gain/permutation importance still available

**Issue**: Class imbalance warnings  
**Solution**: Expected; model uses `scale_pos_weight` to handle imbalance

**Issue**: Missing features  
**Solution**: Model dynamically skips missing columns; ensure feature names match training data

## Future Enhancements

- [ ] Predict Freedom Score trends (churn risk, spending trajectory)
- [ ] Fairness audit by demographic segment
- [ ] Real-time scoring API endpoint
- [ ] Dashboard for monitoring segment distributions over time
- [ ] A/B testing framework for segment-based strategies

## Metadata

- **Model Type**: LightGBM Binary Classification
- **Prediction Target**: P(high-value user)
- **Output Range**: 0–100 (Freedom Score)
- **Segments**: 3 (High/Medium/Low)
- **Features**: ~40 behavioral and ecosystem signals
- **Primary Validation**: 5-fold stratified cross-validation
- **Last Updated**: 2026-05-16

## References & Context

This model is built for the **SAPP Hackathon** and aligns with the fintech ecosystem use case. It handles real-world data challenges (duplicates, imbalance, missing values) and produces interpretable, actionable outputs for customer segmentation and strategy.
