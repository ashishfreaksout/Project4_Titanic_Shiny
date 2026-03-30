library(shiny)
library(dplyr)
library(tidyverse)
library(tidymodels)
library(titanic)
library(discrim) 
library(kknn)
library(ranger)
library(glmnet)
library(C50)
library(klaR)
library(purrr)

# --- 1. Helper Functions ---
# Helper to fit a model safely and return results or NULL if it fails
fit_model_safely <- function(wf, val_split) {
  tryCatch({
    fit_resamples(
      wf,
      resamples = val_split,
      metrics = metric_set(accuracy, roc_auc),
      control = control_resamples(save_pred = TRUE)
    )
  }, error = function(e) {
    message("Model training failed: ", e$message)
    NULL
  })
}

# Helper to construct a new passenger dataframe from inputs
create_passenger_data <- function(input) {
  tibble(
    Pclass = factor(input$in_pclass, levels = c("1", "2", "3")),
    Sex = input$in_sex,
    Age = input$in_age,
    SibSp = input$in_sibsp,
    Parch = input$in_parch,
    Fare = input$in_fare,
    Embarked = input$in_embarked
  )
}

# Helper to validate user inputs returning error messages
validate_inputs <- function(input) {
  errors <- c()
  if (is.na(input$in_age) || input$in_age < 0) errors <- c(errors, "Age must be a valid non-negative number.")
  if (is.na(input$in_sibsp) || input$in_sibsp < 0) errors <- c(errors, "SibSp must be a valid non-negative number.")
  if (is.na(input$in_parch) || input$in_parch < 0) errors <- c(errors, "Parch must be a valid non-negative number.")
  if (is.na(input$in_fare) || input$in_fare < 0) errors <- c(errors, "Fare must be a valid non-negative number.")
  return(errors)
}

# --- 2. Data Loading and Prep ---
raw_titanic <- titanic::titanic_train %>%
  mutate(
    # Convert Survived to a factor as required (1 = Yes, 0 = No)
    Survived = factor(ifelse(Survived == 1, "Yes", "No"), levels = c("No", "Yes")),
    Pclass = factor(Pclass)
  ) %>%
  dplyr::select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  mutate(Embarked = na_if(Embarked, ""))

# Split data - ensure validation split is computationally consistent
set.seed(42)
val_rs <- validation_split(raw_titanic, prop = 0.8, strata = Survived)
train_data <- training(val_rs$splits[[1]])

# Recipe
titanic_rec <- recipe(Survived ~ ., data = train_data) %>%
  step_impute_median(Age, Fare) %>%
  step_impute_mode(Embarked) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# --- 3. Model Specifications ---
models_list <- list(
  Null_Model = null_model() %>% set_engine("parsnip") %>% set_mode("classification"),
  kNN = nearest_neighbor(neighbors = 5) %>% set_engine("kknn") %>% set_mode("classification"),
  Boosted_C50 = boost_tree(trees = 50) %>% set_engine("C5.0") %>% set_mode("classification"),
  RandomForest = rand_forest(trees = 100) %>% set_engine("ranger", importance = "impurity") %>% set_mode("classification"),
  LogisticRegression = logistic_reg(penalty = 0.01, mixture = 0.5) %>% set_engine("glmnet") %>% set_mode("classification"),
  NaiveBayes = naive_Bayes() %>% set_engine("klaR") %>% set_mode("classification")
)

# Pre-bundle model specifications into a list of workflows
wfs <- map(models_list, ~ workflow() %>% add_recipe(titanic_rec) %>% add_model(.x))

# --- 4. UI ---
ui <- fluidPage(
  titlePanel("Titanic Survival Prediction Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      width = 4,
      h4("1. Train Models"),
      p("Click below to train all models. Training does not happen on startup."),
      actionButton("train_btn", "Train Models", class = "btn-primary", width = "100%"),
      hr(),
      h4("2. Predict New Passenger"),
      p("Provide passenger info and predict survival based on the best model."),
      selectInput("in_pclass", "Passenger Class (Pclass)", choices = c("1", "2", "3"), selected = "3"),
      selectInput("in_sex", "Sex", choices = c("male", "female"), selected = "male"),
      numericInput("in_age", "Age", value = 30, min = 0, max = 100),
      numericInput("in_sibsp", "Siblings/Spouses Aboard (SibSp)", value = 0, min = 0, max = 10),
      numericInput("in_parch", "Parents/Children Aboard (Parch)", value = 0, min = 0, max = 10),
      numericInput("in_fare", "Fare", value = 15, min = 0),
      selectInput("in_embarked", "Port of Embarkation (Embarked)", choices = c("C", "Q", "S"), selected = "S"),
      actionButton("predict_btn", "Predict Survival", class = "btn-success", width = "100%")
    ),
    
    mainPanel(
      width = 8,
      tabsetPanel(id = "main_tabs",
        tabPanel("Overview & Data", 
          br(),
          h4("Project Overview"),
          p("This Shiny application uses Tidymodels to train and compare several machine learning models to predict passenger survival on the Titanic."),
          p("We evaluate multiple candidate models locally. To prevent the app from freezing on startup, models are only trained when you click 'Train Models'."),
          hr(),
          h4("Data Preprocessing Summary"),
          verbatimTextOutput("recipe_print"),
          hr(),
          h4("Raw Data Preview"),
          dataTableOutput("raw_data_table")
        ),
        tabPanel("Model Comparison",
          br(),
          h4("Training Status"),
          verbatimTextOutput("train_status"),
          hr(),
          h4("Validation Performance Metrics"),
          tableOutput("metrics_table")
        ),
        tabPanel("Best Model Validation",
          br(),
          h4("Best Model Evaluated (by Accuracy)"),
          verbatimTextOutput("best_model_name"),
          br(),
          fluidRow(
            column(6, h4("Confusion Matrix"), plotOutput("conf_mat_plot", height = "300px")),
            column(6, h4("ROC Curve"), plotOutput("roc_plot", height = "300px"))
          )
        ),
        tabPanel("Prediction Result",
           br(),
           h4("Prediction for Input Passenger"),
           h3(textOutput("prediction_output"), style = "color: #007bff;"),
           br(),
           p("Note: You must click 'Train Models' to evaluate the candidates, then 'Predict Survival' to see this result.", style = "color: gray;")
        )
      )
    )
  )
)

# --- 5. Server ---
server <- function(input, output, session) {
  
  # Ensure standard loading behavior avoids recalculation overhead globally
  output$recipe_print <- renderPrint({ titanic_rec })
  output$raw_data_table <- renderDataTable({ head(raw_titanic, 50) }, options = list(pageLength = 10))
  
  # Reactive state
  rv <- reactiveValues(
    trained = FALSE,
    metrics = NULL,
    best_wf = NULL,
    best_preds = NULL,
    best_name = NULL,
    status = "Models have not been trained yet. Please click 'Train Models' in the sidebar to start."
  )
  
  # Training trigger
  observeEvent(input$train_btn, {
    # Show loading notification
    showNotification("Training models in progress...", id = "train_notif", duration = NULL, type = "message")
    rv$status <- "Training models... Please wait. This may take a minute or two."
    
    # Switch tab to show progress
    updateTabsetPanel(session, inputId = "main_tabs", selected = "Model Comparison")
    
    # Fit each model robustly and safely
    fit_results <- imap(wfs, function(wflow, wf_name) {
      res <- fit_model_safely(wflow, val_rs)
      if (!is.null(res)) {
        res_metrics <- collect_metrics(res) %>% mutate(wflow_id = wf_name)
        res_preds <- collect_predictions(res) %>% mutate(wflow_id = wf_name)
        list(metrics = res_metrics, preds = res_preds, workflow = wflow, wflow_id = wf_name)
      } else {
        NULL
      }
    })
    
    # Remove failed models
    fit_results <- compact(fit_results)
    removeNotification("train_notif")
    
    if (length(fit_results) == 0) {
      rv$status <- "Error: All models failed to train. Please check dependencies (e.g. kknn, C50, ranger)."
      showNotification("All models failed.", type = "error")
      return()
    }
    
    # Collect metrics
    all_metrics <- map_dfr(fit_results, "metrics") %>%
      dplyr::select(wflow_id, .metric, mean) %>%
      pivot_wider(names_from = .metric, values_from = mean) %>%
      arrange(desc(accuracy)) 
      
    rv$metrics <- all_metrics
    
    # Identify the best 
    best_id <- all_metrics$wflow_id[1]
    best_model_info <- keep(fit_results, ~ .x$wflow_id == best_id)[[1]]
    
    # Fit the best model on the complete training set
    tryCatch({
      rv$best_wf <- workflows::fit(best_model_info$workflow, data = train_data)
      rv$best_preds <- best_model_info$preds
      rv$best_name <- best_id
      rv$trained <- TRUE
       
       failed_count <- length(wfs) - length(fit_results)
       if (failed_count > 0) {
         rv$status <- sprintf("Models trained successfully! Note: %d model(s) failed and were skipped.", failed_count)
         showNotification("Note: Some models failed to train.", type = "warning", duration = 5)
       } else {
         rv$status <- "All models trained successfully!"
         showNotification("Models trained successfully!", type = "message", duration = 3)
       }
    }, error = function(e) {
       rv$status <- paste("Failed to refit best model:", e$message)
       showNotification("Failed to refit best model.", type = "error")
    })
  })
  
  output$train_status <- renderText({ rv$status })
  
  output$metrics_table <- renderTable({
    req(rv$trained)
    rv$metrics
  })
  
  output$best_model_name <- renderText({
    req(rv$trained)
    paste("The best model based on accuracy is:", rv$best_name)
  })
  
  output$conf_mat_plot <- renderPlot({
    req(rv$trained, rv$best_preds)
    rv$best_preds %>% 
      conf_mat(truth = Survived, estimate = .pred_class) %>%
      autoplot(type = "heatmap") + 
      labs(title = "Confusion Matrix", subtitle = "Validation Set")
  })
  
  output$roc_plot <- renderPlot({
    req(rv$trained, rv$best_preds)
    if (".pred_Yes" %in% colnames(rv$best_preds)) {
      rv$best_preds %>%
        roc_curve(truth = Survived, .pred_Yes) %>%
        autoplot() +
        labs(title = paste("ROC Curve for", rv$best_name))
    } else {
      plot(1, type = "n", main = "ROC Curve not available for this model", xlab="", ylab="", axes=FALSE)
    }
  })
  
  # Make predictions
  output$prediction_output <- renderText({ "Awaiting input..." })
  
  observeEvent(input$predict_btn, {
    if (!rv$trained) {
       showNotification("Please train the models first before predicting in the sidebar.", type = "warning", duration = 3)
       return()
    }
    
    # Validation messages
    errs <- validate_inputs(input)
    if (length(errs) > 0) {
      showNotification(paste(errs, collapse = "\n"), type = "error", duration = 5)
      return()
    }
    
    new_data <- create_passenger_data(input)
    
    tryCatch({
      pred <- predict(rv$best_wf, new_data = new_data)
      output$prediction_output <- renderText({
        paste("Predicted Survival:", as.character(pred$.pred_class))
      })
      updateTabsetPanel(session, inputId = "main_tabs", selected = "Prediction Result")
    }, error = function(e) {
      output$prediction_output <- renderText({
        paste("Error during prediction:", e$message)
      })
    })
  })
}

shinyApp(ui = ui, server = server)
