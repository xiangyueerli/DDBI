#%% md
# # Lecture 4: Machine Learning Methods for Multivariate Time Series Forecasting
# #
# **Goals of This Lecture:**
# #
# In the previous lectures, you explored various time series analysis and forecasting techniques—from basic univariate forecasting methods to more complex multivariate approaches like VAR. In this lecture, you will apply several **machine learning** models to forecast a target variable (ENSO) using multiple regressors (AAO, AO, NAO, PNA, ENSO itself as a lagged predictor if desired).
# #
# Specifically, you will:
# - **Section 1**: Introduce **Support Vector Regression (SVR)** for multivariate forecasting.
# - **Section 2**: Repeat the process using **K-Nearest Neighbors** regression.
# - **Section 3**: Use a **Decision Tree** regressor.
# - **Section 4**: Use a **Random Forest** regressor.
# - **Section 5**: Compare models in two ways:
#   1. **Simple Out-of-Sample Forecast**: Fit the model once on the training set and forecast the entire test set.
#   2. **Rolling Forecast**: Fit the model in a rolling (iterative) manner, forecasting one step at a time and then updating the training data.
# #
# You may find that each model comes with its own pros and cons. You will discuss them along the way.
# #
# As always, you will use a second-person narrative (“You may notice… You need to…”) to guide you through the learning process.
# 
#%% md
# ## SECTION 0: Imports and Helper Functions
# #
# You first need to import the necessary libraries and define helper functions such as error metrics and confidence-interval calculators.
# 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
sns.set()
plt.style.use('ggplot')

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def max_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def mase(y_true, y_pred, y_train):
    numerator = np.mean(np.abs(y_true - y_pred))
    denominator = np.mean(np.abs(np.diff(y_train)))
    return numerator / denominator

def bootstrap_confidence_intervals(predictions, residuals, alpha=0.1):
    # A simple CI approximation using std of residuals and a z-score ~1.645 for ~90% CI
    if len(residuals) == 0:
        return predictions, predictions
    std_resid = np.std(residuals)
    z_value = 1.645
    lower = predictions - z_value * std_resid
    upper = predictions + z_value * std_resid
    return lower, upper


#%% md
# ## SECTION 1: Support Vector Regression (SVR) for Multivariate Forecasting
# #
# In this section, you will:
# 1. **Load** the `combined_climate_indices_2024.csv` dataset (with columns `Date, AAO, AO, NAO, PNA, ENSO`).
# 2. **Visualize** the time series.
# 3. Prepare the data for **multivariate forecasting** of `ENSO`.
# 4. **Train/Test split**.
# 5. Fit an **SVR** model to the **training** set (in-sample fit).
# 6. Produce an **out-of-sample** forecast for the **test** set.
# 7. **Visualize** everything on one plot:
#    - The entire time series (train + test)
#    - A vertical line at the train-test split
#    - The in-sample fitted values
#    - The out-of-sample forecast (with dashed line) and **90% CI** bands.
# #
# **Brief Introduction to SVR**:
# - You may find that Support Vector Regression uses the concept of support vectors from SVM (Support Vector Machines).
# - It attempts to fit a function within a certain threshold (epsilon) of your data points, balancing complexity with error tolerance.
# - **Pros**:
#   - Handles high-dimensional data well and can model non-linear relationships (using kernels).
#   - Often robust to outliers if you choose parameters carefully.
# - **Cons**:
#   - Choosing the right kernel and hyperparameters (C, epsilon, gamma) can be tricky.
#   - SVR can be slow for very large datasets, as it can scale poorly with data size.
# 
#%%
# --------------------------
# Step 1: Load and prepare data
# --------------------------
df = pd.read_csv('combined_climate_indices_2024.csv', parse_dates=['Date'], dayfirst=False)
df = df.set_index('Date').sort_index()
df = df.asfreq('MS')  # monthly start frequency
df.interpolate(method='linear', inplace=True)

# Let's define the regressors (X) and the target (y)
target_col = 'ENSO'
X = df.drop(columns=[target_col])  # All columns except ENSO
y = df[target_col]

# --------------------------
# Visualization
# --------------------------
plt.figure(figsize=(12, 5))
for col in df.columns:
    plt.plot(df.index, df[col], label=col)
plt.title("Multivariate Climate Indices Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()

df = pd.read_csv('combined_climate_indices_2024.csv', parse_dates=['Date'], dayfirst=False)
df = df.set_index('Date').sort_index()
df = df.asfreq('MS')
df.interpolate(method='linear', inplace=True)

target_col = 'ENSO'
X = df.drop(columns=[target_col])
y = df[target_col]

# ### 1B. Zoomed Plot (Approx. 2015 - 2025)

zoom_start = '2020-01-01'
zoom_end = '2025-12-01'
df_zoom = df.loc[zoom_start:zoom_end]

plt.figure(figsize=(12, 5))
for col in df_zoom.columns:
    plt.plot(df_zoom.index, df_zoom[col], label=col)
plt.title("Multivariate Climate Indices (Zoomed from 2015 to 2025)")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.show()

# --------------------------
# Step 2: Train/test split
# --------------------------
train_ratio = 0.8
n = len(df)
split_index = int(n * train_ratio)

train_X = X.iloc[:split_index]
train_y = y.iloc[:split_index]
test_X = X.iloc[split_index:]
test_y = y.iloc[split_index:]

# --------------------------
# Step 3: Fit various SVR models (different kernels) on training data
# --------------------------
# We'll demonstrate three kernels for variety: 'linear', 'rbf', and 'poly'.
svr_models = {
    'SVR_Linear': SVR(kernel='linear', C=1.0, epsilon=0.1),
    'SVR_RBF': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'SVR_Poly': SVR(kernel='poly', C=1.0, epsilon=0.1, degree=2),
    'SVR_Sigmoid': SVR(kernel='sigmoid', C=1.0, epsilon=0.1),
    'SVR_RBF_Tweaked': SVR(kernel='rbf', C=10.0, gamma=0.1, epsilon=0.1)
}

results_svr = {}

for model_name, svr_model in svr_models.items():
    svr_model.fit(train_X, train_y)
    # In-sample fitted values
    in_sample_fit = svr_model.predict(train_X)
    # Out-of-sample forecast
    out_sample_pred = svr_model.predict(test_X)

    results_svr[model_name] = {
        'model': svr_model,
        'in_sample_fit': in_sample_fit,
        'out_sample_pred': out_sample_pred
    }

# --------------------------
# Step 4: Visualize for one chosen SVR model, e.g. RBF
# --------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()  # To index them easily in a loop

# We'll iterate over the five models you trained
model_names = list(results_svr.keys())  # e.g. ['SVR_Linear', 'SVR_RBF', 'SVR_Poly', 'SVR_Sigmoid', 'SVR_RBF_Tweaked']

for i, model_name in enumerate(model_names):
    ax = axes[i]

    # Retrieve the in-sample fit and out-of-sample predictions
    in_sample_fit = results_svr[model_name]['in_sample_fit']
    out_sample_pred = results_svr[model_name]['out_sample_pred']

    # Calculate residuals on training set for CI
    train_residuals = train_y.values - in_sample_fit
    lower_ci, upper_ci = bootstrap_confidence_intervals(out_sample_pred, train_residuals)

    # Plot the observed series
    ax.plot(y.index, y, label='Observed ENSO', color='black')
    # Train-test split line
    ax.axvline(x=y.index[split_index], color='orange', linestyle='--', label='Train-Test Split')
    # In-sample fit
    ax.plot(train_y.index, in_sample_fit, label=f'{model_name} In-sample', color='blue')
    # Out-of-sample forecast
    ax.plot(test_y.index, out_sample_pred, label=f'{model_name} Forecast', color='blue', linestyle='--')
    # Confidence bands
    ax.fill_between(test_y.index, lower_ci, upper_ci, color='blue', alpha=0.1, label='90% CI')

    ax.set_title(f"SVR Forecast ({model_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("ENSO Value")
    ax.legend()

# Hide the unused 6th subplot (if you have only 5 models)
axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

#%% md
# **Discussion of SVR Results:**
# - You may find that the linear kernel might underfit if the relationship is non-linear.
# - The RBF kernel often captures non-linear relationships better, but it requires tuning the hyperparameters like `C` and `gamma`.
# - The polynomial kernel can also capture non-linearity, but it can become complex quickly if the degree is large.
# - Look at the residuals and forecast accuracy to decide which kernel might be best.
# - Always consider a grid search or cross-validation for hyperparameter tuning in practice.
# 
#%% md
# ## SECTION 2: K-Nearest Neighbors Regression
# #
# **Why KNN?**
# - K-Nearest Neighbors (KNN) regression predicts a new data point’s value by averaging the values of its K nearest neighbors in the feature space.
# - You may find it easy to understand and intuitive.
# - **Pros**:
#   - Non-parametric: no assumption about data distribution.
#   - Simple conceptually.
# - **Cons**:
#   - Can be slow for large datasets (needs to search nearest neighbors).
#   - Sensitive to scaling of features and choice of K.
# #
# **Process**:
# 1. Use the same train/test split as before.
# 2. Fit KNN on the training set.
# 3. Visualize in-sample fit and out-of-sample forecast with the same approach (90% CI).
# 4. Discuss results, pros, and cons.
# #
# **Statistical Note**:
# - KNN doesn’t build an explicit “model” in the usual sense; it stores the training data and queries it during prediction.
# - You need to remember that if your data has many features, distance-based methods like KNN can suffer from the curse of dimensionality.
# 
#%%
# We'll use the same train_X, train_y, test_X, test_y
knn_model = KNeighborsRegressor(n_neighbors=5)  # You may vary n_neighbors
knn_model.fit(train_X, train_y)

knn_in_sample_fit = knn_model.predict(train_X)
knn_out_sample_pred = knn_model.predict(test_X)

# Residuals for CI from training set
knn_residuals = train_y.values - knn_in_sample_fit
knn_lower_ci, knn_upper_ci = bootstrap_confidence_intervals(knn_out_sample_pred, knn_residuals)

# Visualization
plt.figure(figsize=(12, 5))
plt.plot(y.index, y, label='Observed ENSO', color='black')
plt.axvline(x=y.index[split_index], color='orange', linestyle='--', label='Train-Test Split')

# In-sample fit
plt.plot(train_y.index, knn_in_sample_fit, label='KNN In-sample', color='red')

# Out-of-sample forecast
plt.plot(test_y.index, knn_out_sample_pred, label='KNN Forecast', color='red', linestyle='--')
plt.fill_between(test_y.index, knn_lower_ci, knn_upper_ci, color='red', alpha=0.1, label='90% CI')

plt.title("K-Nearest Neighbors Regression Forecast")
plt.xlabel("Date")
plt.ylabel("ENSO Value")
plt.legend()
plt.show()

#%% md
# **Discussion of KNN:**
# - You may find that the forecast can look quite choppy since it is based on local neighbors.
# - If `K` is too small, the model can be noisy. If `K` is too large, it can over-smooth.
# - Scaling your features (e.g., using StandardScaler or MinMaxScaler) is often important for KNN because distance metrics can become skewed otherwise.
# - KNN has no built-in mechanism to handle time-lag dependencies directly (unless you engineer features for it).
# 
#%% md
# ## SECTION 3: Decision Tree Regression
# #
# **Why Decision Trees?**
# - A Decision Tree splits the feature space into regions based on specific conditions (if X < a, go left, else go right, etc.).
# - **Pros**:
#   - Very interpretable (you can see the tree structure).
#   - Handles non-linear relationships automatically.
# - **Cons**:
#   - Can overfit easily if you don’t tune parameters (like `max_depth`).
#   - High variance—small changes in data can lead to different splits.
# #
# **Process**:
# 1. Use the same train/test split.
# 2. Fit the Decision Tree (with minimal parameter tuning).
# 3. Plot in-sample, out-of-sample, and 90% CI.
# 
#%%
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)  # You may adjust depth
tree_model.fit(train_X, train_y)

tree_in_sample_fit = tree_model.predict(train_X)
tree_out_sample_pred = tree_model.predict(test_X)

tree_residuals = train_y.values - tree_in_sample_fit
tree_lower_ci, tree_upper_ci = bootstrap_confidence_intervals(tree_out_sample_pred, tree_residuals)

# Visualization
plt.figure(figsize=(12, 5))
plt.plot(y.index, y, label='Observed ENSO', color='black')
plt.axvline(x=y.index[split_index], color='orange', linestyle='--', label='Train-Test Split')

plt.plot(train_y.index, tree_in_sample_fit, label='Decision Tree In-sample', color='green')
plt.plot(test_y.index, tree_out_sample_pred, label='Decision Tree Forecast', color='green', linestyle='--')
plt.fill_between(test_y.index, tree_lower_ci, tree_upper_ci, color='green', alpha=0.1, label='90% CI')

plt.title("Decision Tree Regression Forecast")
plt.xlabel("Date")
plt.ylabel("ENSO Value")
plt.legend()
plt.show()

#%% md
# **Discussion of Decision Trees:**
# - You may find that a shallow tree (small max_depth) might underfit, while a deep tree might overfit.
# - Decision Trees can model interactions between features but can be unstable if not pruned or tuned.
# - For time-series forecasting, you often need to consider feature engineering (e.g., lags, rolling statistics) if your data has strong temporal structure.
# 
#%% md
# ## SECTION 4: Random Forest Regression
# #
# **Why Random Forest?**
# - A Random Forest is an ensemble of decision trees where each tree is trained on a random subset of data and features.
# - **Pros**:
#   - Often improves forecast stability and accuracy compared to a single Decision Tree.
#   - Handles non-linearities and interactions quite well.
# - **Cons**:
#   - Less interpretable than a single tree.
#   - Can still overfit if not tuned carefully.
# #
# **Process**:
# 1. Same train/test split.
# 2. Fit the Random Forest on the training set.
# 3. Visualize the in-sample predictions, out-of-sample forecast, and 90% CI.
# 
#%%
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(train_X, train_y)

rf_in_sample_fit = rf_model.predict(train_X)
rf_out_sample_pred = rf_model.predict(test_X)

rf_residuals = train_y.values - rf_in_sample_fit
rf_lower_ci, rf_upper_ci = bootstrap_confidence_intervals(rf_out_sample_pred, rf_residuals)

plt.figure(figsize=(12, 5))
plt.plot(y.index, y, label='Observed ENSO', color='black')
plt.axvline(x=y.index[split_index], color='orange', linestyle='--', label='Train-Test Split')

plt.plot(train_y.index, rf_in_sample_fit, label='Random Forest In-sample', color='purple')
plt.plot(test_y.index, rf_out_sample_pred, label='Random Forest Forecast', color='purple', linestyle='--')
plt.fill_between(test_y.index, rf_lower_ci, rf_upper_ci, color='purple', alpha=0.1, label='90% CI')

plt.title("Random Forest Regression Forecast")
plt.xlabel("Date")
plt.ylabel("ENSO Value")
plt.legend()
plt.show()

#%% md
# **Discussion of Random Forest:**
# - You may find that the ensemble approach reduces variance (less overfitting) compared to a single Decision Tree.
# - Tuning parameters such as `n_estimators`, `max_depth`, and `min_samples_split` can greatly influence performance.
# - It might handle non-linearities in the climate indices better, especially if there are complex interactions.
# 
#%% md
# ## SECTION 5: Accuracy Comparison for Simple vs. Rolling Forecasts
# #
# You now have four ML models:
# 1. **SVR** (choose one variant, e.g. RBF, or compare all)
# 2. **KNN**
# 3. **Decision Tree**
# 4. **Random Forest**
# #
# You will:
# 1. Compute accuracy metrics (max_error, MAE, MSE, RMSE, MAPE, MASE) for each model’s **simple out-of-sample forecast**.
# 2. Implement a **rolling forecast** approach for each model and again compute metrics.
# #
# ### 5.1 Simple Out-of-Sample Forecast Comparison
# 
#%%
# Let's store predictions from each model in a dictionary.
# You already have them from the code above. We'll pick one SVR variant (RBF) for demonstration.
svr_rbf_out = results_svr['SVR_RBF']['out_sample_pred']
knn_out = knn_out_sample_pred
tree_out = tree_out_sample_pred
rf_out = rf_out_sample_pred

test_actual = test_y.values  # True values in test set
train_actual = train_y.values


# We'll define a function to compute the metrics in one go
def compute_metrics(y_true, y_pred, y_train):
    return {
        'max_error': max_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'MASE': mase(y_true, y_pred, y_train)
    }


models_simple = {
    'SVR_RBF': svr_rbf_out,
    'KNN': knn_out,
    'DecisionTree': tree_out,
    'RandomForest': rf_out
}

simple_results = {}
for m_name, preds in models_simple.items():
    simple_results[m_name] = compute_metrics(test_actual, preds, train_actual)

df_simple_results = pd.DataFrame(simple_results).T
print("Simple Out-of-Sample Forecasting Accuracy (Metrics):")
print(df_simple_results)


#%% md
# ### 5.2 Rolling Forecast
# #
# **Rolling Forecast Steps**:
# 1. Start with the training set.
# 2. Fit model, predict the next step (one-step ahead) in the test set.
# 3. Append the actual observation from the test set to the training data (“roll” forward).
# 4. Refit or update the model, predict the next step.
# 5. Repeat until you forecast the entire test set, one step at a time.
# #
# This is computationally more expensive but more realistic.
# 
#%%
def rolling_forecast(model_constructor, train_X, train_y, test_X, test_y):
    """
    Generic rolling forecast function:
    - model_constructor: a function or constructor that returns a fresh model instance
    - train_X, train_y: initial training data
    - test_X, test_y: the entire test set
    You will do 1-step ahead forecast iteratively.
    """
    history_X = train_X.copy()
    history_y = train_y.copy()
    preds = []

    for i in range(len(test_X)):
        # Instantiate and fit a new model
        model = model_constructor()
        model.fit(history_X, history_y)

        # Predict the next step
        x_next = test_X.iloc[i:i + 1]  # single step
        pred_next = model.predict(x_next)[0]
        preds.append(pred_next)

        # "Roll" forward: add the actual observation to the history
        y_next = test_y.iloc[i]  # the true value
        history_X = pd.concat([history_X, x_next])
        history_y = pd.concat([history_y, pd.Series([y_next], index=[test_y.index[i]])])

    return np.array(preds)


# We'll define constructors for each model we want to test in rolling mode

def svr_rbf_constructor():
    return SVR(kernel='rbf', C=1.0, epsilon=0.1)


def knn_constructor():
    return KNeighborsRegressor(n_neighbors=5)


def tree_constructor():
    return DecisionTreeRegressor(max_depth=5, random_state=42)


def rf_constructor():
    return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)


# Now let's do rolling forecasts
svr_rbf_rolling = rolling_forecast(svr_rbf_constructor, train_X, train_y, test_X, test_y)
knn_rolling = rolling_forecast(knn_constructor, train_X, train_y, test_X, test_y)
tree_rolling = rolling_forecast(tree_constructor, train_X, train_y, test_X, test_y)
rf_rolling = rolling_forecast(rf_constructor, train_X, train_y, test_X, test_y)

models_rolling = {
    'SVR_RBF': svr_rbf_rolling,
    'KNN': knn_rolling,
    'DecisionTree': tree_rolling,
    'RandomForest': rf_rolling
}

rolling_results = {}
for m_name, preds in models_rolling.items():
    rolling_results[m_name] = compute_metrics(test_actual, preds, train_actual)

df_rolling_results = pd.DataFrame(rolling_results).T
print("\nRolling Forecasting Accuracy (Metrics):")
print(df_rolling_results)

#%% md
# **Interpretation:**
# - Compare the simple forecast metrics vs. the rolling forecast metrics.
# - You may find that rolling forecasts can sometimes yield different (often more conservative) error values, as the model is constantly being retrained.
# - The best model is not always obvious; it may depend on hyperparameters and data characteristics.
# #
# **General Pros and Cons of These ML Models**:
# - **SVR**: Flexible with kernels, can model non-linearities, but parameter tuning is crucial.
# - **KNN**: Easy to understand, no explicit model training, but can be slow and sensitive to the choice of `K` and feature scaling.
# - **Decision Tree**: Interpretable, can handle complex data, but can overfit if not pruned.
# - **Random Forest**: Often more robust and accurate than a single tree, but less interpretable and can still overfit if not carefully tuned.
# #
# **Practical Insights**:
# - Always perform feature engineering (e.g., lags, rolling means) if your time series has strong temporal dependencies.
# - Consider cross-validation or grid search to find optimal hyperparameters.
# - Evaluate models using multiple metrics and consider rolling forecasts for a more realistic scenario.
# #
# # Summary & Next Steps
# #
# In this lecture, you have:
# 1. Explored **Support Vector Regression** with different kernels to forecast ENSO.
# 2. Tried a **K-Nearest Neighbors** approach, highlighting its reliance on distance metrics.
# 3. Used a **Decision Tree Regressor**, noting the risk of overfitting if the tree is too deep.
# 4. Applied a **Random Forest Regressor**, which ensembles multiple trees and often provides better stability.
# 5. Compared all models using **simple out-of-sample** and **rolling** forecasts through various error metrics.
# #
# **Key Takeaways**:
# - Machine Learning models can capture non-linearities but require careful hyperparameter tuning.
# - You need to remember that time series often need specialized feature engineering to capture lags and seasonality effectively.
# - Rolling forecasts present a more realistic scenario but at higher computational cost.
# #
# Moving forward, you might explore even more advanced models (e.g., Gradient Boosting, Neural Networks, or Hybrid Approaches) and more sophisticated time series feature engineering. Always keep an eye on the assumptions, data size, and interpretability when selecting your model.
# 