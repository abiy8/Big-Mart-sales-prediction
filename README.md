# Big Mart Sales Prediction

Predict item-level retail sales from product and outlet attributes using an XGBoost regression model. This notebook project covers data exploration, missing-value handling, categorical encoding, and train/test evaluation.

**Stack:** Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn

## Workflow

1. Load the included `Train.csv` and inspect product/outlet features.
2. Explore distributions and handle missing values.
3. Encode categorical variables and separate `Item_Outlet_Sales` as the target.
4. Use an 80/20 train/test split (`random_state=2`).
5. Train `XGBRegressor` and calculate training and test R².

## Run locally

```bash
git clone https://github.com/abiy8/Big-Mart-sales-prediction.git
cd Big-Mart-sales-prediction
python -m venv .venv
# Activate .venv for your operating system.
pip install jupyter pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook
```

Open `big mart sales prediction .ipynb` and run cells in order from the repository directory. `Test.csv` is also included.

## Evaluation and limitations

The notebook prints R² scores; saved outputs are exploratory results, not an independently reproduced performance claim. Preprocessing currently occurs before the split, so a future evaluation should fit preprocessing on training data only using a pipeline and cross-validation. This is a learning project, not a deployed sales forecasting service.
