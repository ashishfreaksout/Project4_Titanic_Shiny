# Project : Titanic Survival Prediction Dashboard

This repository contains a full Shiny web application built using the R `shiny`, `tidymodels`, and `tidyverse` frameworks. The dashboard interactively trains, evaluates, and compares multiple machine learning models to predict passenger survival on the Titanic dataset, allowing the user to make real-time predictions for custom passengers.

## Project Files

* **`app.R`**: The core application logic. This script contains the full UI and Server architecture, data preprocessing (handling missing values via `recipes`), the model configurations, and the robust validation mapping that evaluates candidates based on validation accuracy and ROC AUC.
* **`report.qmd`**: A comprehensive Quarto report documenting the methodology, the Five Steps of data analysis, model comparison outcomes, and details on how agentic programming methodologies were utilized during development.
* **`screenshots/`**: Directory containing screenshots of the dashboard UI stages.

## Models Evaluated

The application utilizes the `titanic::titanic_train` dataset and evaluates the following specifications against a 20% validation split:
1. Null Model
2. k-Nearest Neighbors (kNN)
3. Boosted C5.0
4. Random Forest (Selected natively by the app)
5. Regularized Logistic Regression
6. Naive Bayes

## Installation & Usage

To run this application locally, you must have the following dependencies installed in your R environment:

```r
install.packages(c("shiny", "tidyverse", "tidymodels", "titanic", "discrim", "kknn", "ranger", "glmnet", "C50", "klaR", "purrr"))
```

After dependencies are verified, simply run:

```r
# From inside the project directory in R or RStudio
shiny::runApp("app.R")
```

1. Navigate to the **Data Preparation** tab to see the preprocessing rules.
2. Visit the **Model Comparison** tab and trigger the training mechanism. 
3. Observe the metric evaluations and navigate to the **Best Model** tab to evaluate performance constraints (including ROC Curves and Confusion Matrices).
4. Provide hypothetical demographic information in the **Predict New Passenger** section to yield real-time survival predictions! 
