
# Designing lung cancer screening programs with machine learning
#### CPH 100A Problem Set 1
#### Due Date: 10:00 AM PT, Fri Sept 11, 2026
#### Everyone is required to submit their own code and their own report.

## Introduction

Lung cancer screening with low-dose computed tomography significantly improves patient lung cancer outcomes, improving survival and reducing morbidity; two large randomized control lung cancer screening trials have demonstrated 20% (NLST trial) and 24% (NELSON trial) reductions in lung cancer mortality respectively. These results have motivated the development of national lung screening programs. The success of these programs hinges on their ability to screen the right patients, balancing the benefits of early detection against the harms of overscreening. This capacity relies on our ability to estimate a patient's risk of developing lung cancer. In this problem set, we will develop machine learning tools to predict lung cancer risk from PLCO questionnaires, develop screening guideline simulations, and compare the cost-effectiveness of these proposed guidelines against current NLST criteria. The goal of this problem set is to give you hands-on experience developing machine learning tools from scratch and analyzing their clinical implications in a real world setting. At the end of this problem set, you will write a short report, describing your model and your analyses. Submit your **problem set code** and a **report** by the due date.

## Deliverables

There are no teams in this problem set. Every student submits their own code
and their own report, as two separate submissions.

### Code submission (Part 1)
- Complete implementation of `logistic_regression.py`, `vectorizer.py`, and `dispatcher.py`
- Working `main.py` that prints the model AUC on the test set
- Grid search results CSV file

### Report submission (Part 2)
- A short report covering Parts 2.1-2.5, including your subgroup performance
  analysis and clinical utility simulation results

**Both due**: 10:00 AM PT, Fri Sept 11, 2026

### Collaboration policy
You are welcome to work with your classmates, and we encourage you to. Talk
through the concepts, debug together, compare modeling choices, and help each
other past sticking points. 

What you hand in has to be your own. Write your own code and your own report,
and note at the top of your submission who you worked with.


### Use of AI in this problem set
- **Individual Implementation**: For your own learning, we encourage you NOT to use AI to solve the core model implementation (logistic regression, gradient computation, age vectorizer). Going through these manually will help you master gradient descent and SGD concepts. Discuss challenges with classmates and instructors at office hours.
- **Encouraged AI Use**: You are explicitly encouraged to use AI tools for the full vectorizer, dispatcher, and the Part 2 analyses. Use LLMs for report feedback and any other helpful tasks.

Remember: the key goal is your learning, not top performance. Use AI tools to boost your mastery, not replace learning opportunities.

## Part 0: Setup

### Installation and Environment

**Step 1: Install Miniconda**
Download and install Miniconda from https://docs.conda.io/en/latest/miniconda.html
- Choose Python 3.10+ for your OS
- Follow installer defaults

**Step 2: Create and activate environment**
```bash
# From the pset1 directory
conda create -n cph100_project1 python=3.10 -y
conda activate cph100_project1

# Install dependencies
pip install -r requirements.txt
```

**Step 3: Verify Installation**
```bash
python check_installation.py
```

**Troubleshooting:**
- Out of space: clean conda cache: `conda clean --all`
- If conda is unavailable: use system Python 3.10+ and run `pip install -r requirements.txt`
- Still having issues: ask LLMs, classmates, or come to office hours


The PLCO dataset files, including a helpful data dictionaries and the raw data csv, are available on bCourses in the "pset1_data" folder in "Files". If you are completing this assignment without being officially registered for the class, email Irene and she will send you the data directly.

### Code Structure Overview
- `main.py`: Main training loop (modify feature_config here)
- `vectorizer.py`: Data preprocessing (implement all TODO functions)  
- `logistic_regression.py`: Model implementation (implement fit, gradient, predict methods)
- `dispatcher.py`: Hyperparameter search (implement get_experiment_list, launch_experiment)

## Part 1: Model Development - Code Submission [50 pts]

In this part of the problem set, you will extend the starter code to develop lung cancer risk models from the PLCO data.

### 1.1: Implementing a simple age-based Logistic Regression Classifier

To get started, we will implement logistic regression with Stochastic Gradient Descent to predict lung cancer risk using just patient age.

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
3. Derive and implement the gradient equations in `logistic_regression.py`. Note, you may way to review the lecture slides.
    - Recall the stability trick noted in lecture.
4. Implement SGD training loop

**Expected Performance:**
Your age-only validation AUC should be approximately **0.60**. This is reasonable because age is a strong predictor of lung cancer risk, but limited on its own.

### 1.2: Implementing a grid search dispatcher

A key challenge in developing effective machine learning tools is experiment management. In this section, you'll develop a job dispatcher for hyperparameter grid search.
Note, you are encouraged to use AI for advice on this section.

**Requirements:**
- Take hyperparameter configurations from `grid_search.json`
- Run experiments for each parameter combination
- Track and summarize results in CSV format
- Support parallel execution with multiple workers

**Deliverable:**
Complete the grid search dispatcher and use it to tune your age-based model. In your report, include a plot showing the relationship between L2 regularization and training loss.

### 1.3: Building your best PLCO risk model

Now extend your model to include additional features from the PLCO dataset.

**Feature Engineering Steps:**
1. **Data Exploration**: Explore the dataset using the data dictionary
2. **Feature Selection**: Choose 5-10 meaningful features beyond age (suggestions: smoking history, family history, demographics)
3. **Vectorization Implementation**: Handle different data types:
   - **Numerical features** (age, pack-years): normalize to zero mean, unit variance
   - **Categorical features** (sex, race): one-hot encoding  
   - **Ordinal features** (education): integer encoding
4. **Missing Data Strategy**: Implement mean imputation, indicator variables, or other approach

**Advanced Models** (Optional):
Consider implementing Random Forest or Gradient Boosted Trees using `sklearn` for comparison.

**Expected Performance:**
Your final validation AUC should be ≥ **0.80**. This target is achievable with good feature engineering and proper hyperparameter tuning.

**Common Debugging Issues:**
- **Model not converging**: Try lower learning rate or more epochs
- **Poor performance**: Check feature normalization and missing data handling
- **Gradient explosion**: Verify gradient computation

## Part 2: Evaluation - Report Submission [50 pts]
Converge on a single model implementation and perform detailed analysis of model performance. You are encouraged to use AI to help you in this section.

### 2.1: Ablation study
Include training and validation loss curves and describe your best model implementation details. What design decisions were most important for achieving good performance?

### 2.2: Analyzing overall model performance 
Evaluate your model on the test set and various subgroups:

**Required Plots:**
- ROC curves and Precision-Recall curves for your best model
- Highlight the operating point of current NLST criteria (`"nlst_flag"` column)

**Subgroup Analysis** (report AUC for each):
- Sex (`sex` column)
- Race (`race7` column)  
- Educational status (`educat` column)
- Cigarette smoking status (`cig_stat` column)
- NLST eligibility (`nlst_flag` column)

**Analysis Questions:**
Are there AUC differences >0.05 between groups? What might cause these disparities? What are the limitations of these analyses? 

### 2.3: Model interpretation
List your top 3 most important features and explain how you identified them?

### 2.4: Simulating Clinical Utility
Compare your model against NLST criteria by computing:

1. **NLST Baseline**: Sensitivity, specificity, and PPV of NLST criteria on PLCO test set
2. **Matched Performance**: If you match NLST's specificity, what sensitivity does your model achieve?
3. **Risk Threshold Selection**: How would you choose a screening threshold? Justify your choice.
4. **Subgroup Performance**: Compute metrics for your chosen threshold across all patient subgroups

### 2.5: Identifying limitations in study design
Discuss implications and limitations: What's missing from these analyses? What additional studies are needed to inform screening guidelines?
