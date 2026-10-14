
# Designing lung cancer screening programs with machine learning

Lung cancer screening with low-dose computed tomography significantly improves patient lung cancer outcomes, improving survival and reducing morbidity; two large randomized control lung cancer screening trials have demonstrated 20% (NLST trial) and 24% (NELSON trial) reductions in lung cancer mortality respectively. These results have motivated the development of national lung screening programs. The success of these programs hinges on their ability to screen the right patients, balancing the benefits of early detection against the harms of overscreening. This capacity relies on the ability to estimate a patient's risk of developing lung cancer. Goal: develop machine learning tools to predict lung cancer risk from PLCO questionnaires, develop screening guideline simulations, and compare the cost-effectiveness of these proposed guidelines against current NLST criteria. 


### Code Structure Overview
- `main.py`: Main training loop (modify feature_config here)
- `vectorizer.py`: Data preprocessing (implement all TODO functions)  
- `logistic_regression.py`: Model implementation (implement fit, gradient, predict methods)
- `dispatcher.py`: Hyperparameter search (implement get_experiment_list, launch_experiment)


### 1.1: Implementing a simple age-based Logistic Regression Classifier

Started off with implementing logistic regression with Stochastic Gradient Descent to predict lung cancer risk using just patient age.

**Model Definition:**
$$p = \sigma(\theta x + b)$$

where $\theta$ and $b$ are our model parameters, $x$ is our feature vector, and $\sigma$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).

**Loss Function:**
We train our model using binary cross entropy loss with L2 regularization:
$$L(y, p) = -[y \log(p) + (1-y)\log(1-p)] + \frac{\lambda}{2} ||\theta||^2$$

where $y$ is the true label, $p$ is the predicted probability, and $\lambda$ is the regularization parameter.

**Implementation Steps:**
1. Extract age data (column name: `"age"`) from the PLCO CSV
2. Implement feature normalization in `vectorizer.py`
3. Derive and implement the gradient equations in `logistic_regression.py`.
4. Implement SGD training loop

**Expected Performance:**
Your age-only validation AUC should be approximately **0.60**. This is reasonable because age is a strong predictor of lung cancer risk, but limited on its own.

### 1.2: Implementing a grid search dispatcher

A key challenge in developing effective machine learning tools is experiment management. 

**Requirements:**
- Take hyperparameter configurations from `grid_search.json`
- Run experiments for each parameter combination
- Track and summarize results in CSV format
- Support parallel execution with multiple workers


### 1.3: Building your best PLCO risk model

Extended my model to include additional features from the PLCO dataset.

**Feature Engineering Steps:**
1. **Data Exploration**: Explore the dataset using the data dictionary
2. **Feature Selection**: Choose 5-10 meaningful features beyond age (suggestions: smoking history, family history, demographics)
3. **Vectorization Implementation**: Handle different data types:
   - **Numerical features** (age, pack-years): normalize to zero mean, unit variance
   - **Categorical features** (sex, race): one-hot encoding  
   - **Ordinal features** (education): integer encoding
4. **Missing Data Strategy**: Implement mean imputation, indicator variables, or other approach

**Advanced Models**:
Consider implementing Random Forest or Gradient Boosted Trees using `sklearn` for comparison.

**Expected Performance:**
The final validation AUC should be ≥ **0.80**. This target is achievable with good feature engineering and proper hyperparameter tuning.

**Common Debugging Issues:**
- **Model not converging**: Try lower learning rate or more epochs
- **Poor performance**: Check feature normalization and missing data handling
- **Gradient explosion**: Verify gradient computation
4. **Subgroup Performance**: Compute metrics for your chosen threshold across all patient subgroups

### 2.5: Identifying limitations in study design
Discuss implications and limitations: What's missing from these analyses? What additional studies are needed to inform screening guidelines?
