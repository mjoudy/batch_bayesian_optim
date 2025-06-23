# %%
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import xgboost as xgb

# %%
from botorch.utils.sampling import draw_sobol_samples
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

# %% [markdown]
# ✅ Start with initial samples → explore space  
# ✅ Build GP surrogate model  
# ✅ Use acquisition function (qEI) to select new batch  
# ✅ Evaluate batch → update model  
# ✅ Repeat → the GP gets better over time → faster convergence  

# %%
#data = fetch_covtype()
data = fetch_california_housing()
X = data.data
y = data.target

# %%
df = pd.DataFrame(X, columns=data.feature_names)
df['MedHouseVal'] = y
df.head()

# %%
df.info()

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
def train_evaluate(params):
    params = {
        'learning_rate': float(params[0]),
        'max_depth': int(params[1]),
        'min_child_weight': float(params[2]),
        'subsample': float(params[3]),
        'colsample_bytree': float(params[4]),
        'gamma': float(params[5]),
        'lambda': float(params[6]),
        'alpha': float(params[7]),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    # For example, use negative MSE as objective to maximize
    score = -mean_squared_error(y_val, y_pred)
    return score

# %%
# Define bounds for each hyperparameter
bounds = torch.tensor([
    [0.01, 0.3],    # learning_rate
    [3, 10],        # max_depth
    [1, 10],        # min_child_weight
    [0.5, 1.0],     # subsample
    [0.5, 1.0],     # colsample_bytree
    [0, 5],         # gamma
    [0, 5],         # lambda
    [0, 5]          # alpha
], dtype=torch.double).T  # shape (2, d)

# Number of parameters
dim = bounds.shape[0]


# %%
# 2️⃣ Generate initial samples:
n_initial = 20

train_x = draw_sobol_samples(bounds=bounds, n=n_initial, q=1).squeeze(-2)

# 3️⃣ Evaluate objective:
train_obj = torch.tensor([train_evaluate(x) for x in train_x], dtype=torch.double).unsqueeze(-1)

# %%
# Build GP model
model = SingleTaskGP(train_x, train_obj)

# Define Marginal Log-Likelihood (objective for GP training)
mll = ExactMarginalLogLikelihood(model.likelihood, model)

# Fit the GP model to current data
fit_gpytorch_mll(mll)

# %%
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([500]))

# Define acquisition function
qEI = qExpectedImprovement(
    model=model, 
    best_f=train_obj.max(),
    sampler=sampler
)

# %%
# Batch size — how many new points to evaluate per iteration
batch_size = 5

# Optimize acquisition function
candidates, _ = optimize_acqf(
    acq_function=qEI,
    bounds=bounds,
    q=batch_size,
    num_restarts=10,
    raw_samples=100
)


# %%
# Evaluate new candidates
new_obj = torch.tensor([train_evaluate(x) for x in candidates], dtype=torch.double).unsqueeze(-1)

# Update training data
train_x = torch.cat([train_x, candidates], dim=0)
train_obj = torch.cat([train_obj, new_obj], dim=0)

# Refit GP model on updated data
model = SingleTaskGP(train_x, train_obj)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)


# %%
n_iterations = 10
batch_size = 5

for iteration in range(n_iterations):
    print(f"\n=== Iteration {iteration + 1} ===")
    
    # Define acquisition function
    qEI = qExpectedImprovement(
        model=model,
        best_f=train_obj.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500]))

    )
    
    # Optimize acquisition function
    candidates, _ = optimize_acqf(
        acq_function=qEI,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=100
    )
    
    # Evaluate new candidates
    new_obj = torch.tensor([train_evaluate(x) for x in candidates], dtype=torch.double).unsqueeze(-1)
    
    # Update training data
    train_x = torch.cat([train_x, candidates], dim=0)
    train_obj = torch.cat([train_obj, new_obj], dim=0)
    
    # Refit model
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    # Print current best R²
    print(f"Current best R²: {train_obj.max().item():.4f}")


# %%



