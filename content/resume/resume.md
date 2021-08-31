---
# See https://sourcethemes.com/academic/docs/page-builder/
widget: blank
# This file represents a page section.
headless: true
# Activate this widget? true/false
active: true
# Order that this section will appear.
weight: 20

title: ""
subtitle: ""

design:
# Choose how many columns the section has. 1 or 2.
  columns: 1

design.background:
# Apply a background color, gradient, or image.
#   Uncomment (by removing `#`) an option to apply it.  
#   Choose a light or dark text color by setting `text_color_light`.
#   Any HTML color name or Hex value is valid.

# Background color.
# color: "navy"
  
# Background gradient.
# gradient_start: "DeepSkyBlue"
# gradient_end: "SkyBlue"
  
# Background image.
# Name of image in `static/img/`.
# image: ""
# Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.
# image_darken: "0.6"

# Text color (true=light or false=dark).
  text_color_light: true

design.spacing:
# Customize the section spacing. Order is top, right, bottom, left.
  padding: ["100px", "120px", "100px", "120px"]

advanced:
# Custom CSS. 
#css_style: ""
 
# CSS class.
#css_class: ""
---

# Todd Warczak PhD
## My Info {.heading .hide}
White River Junction, VT (Relocating to Seattle, NY, or other major US city)\
206-999-6478\
twarczak@gmail.com\
[github.com/TWarczak](https://github.com/TWarczak)\
[toddwarczak.netlify.app](https://toddwarczak.netlify.app)

## Education {.divider}

### PhD Molecular & Cellular Biology &mdash; Dartmouth College
#### Sep 2012 - August 2020

### BS Biology &mdash; University of Utah
#### Sep 2008 - May 2012

## Work Experience {.divider}

### Molecular Biologist
#### 2012 - 2020
* Engineered genome-wide association study (GWAS) to identify genes controlling arsenic tolerance in plants. Determined Arabidopsis gene NIP1;1 on 4th chromosome as the major genetic factor for tolerating arsenic. All data cleaned/wrangled/analyzed in R.
* Built lab RNA-seq pipeline for gene expression of 25000+ plant genes with R scripts for unsupervised learning (PCA, hierarchical clustering), regression (GLMs/ANOVA), exploratory data analysis, and statistical tests.
* Developed first cell-type specific expression maps for plant genes involved in root arsenic acquisition, efflux, and sequestration (using R).
* Presented research to local community members, government officials, and other stakeholders as a representative of the Dartmouth Toxic Metals Superfund Research Program.
* Mentored 2 undergratuate scientists.

## Recent Projects {.divider}

### SageMaker + RStudio to Predict Home Prices w/ Multi-class XGBoost; Explaining Model Behavior with Geospacial Plots and SHAP

Explored Austin dataset to predict home price in Kaggle competition. ([Blog](https://toddwarczak.netlify.app/post/xgb-multi-class/), [Github](https://github.com/TWarczak/data_warz/blob/master/content/post/2021-08-01-austin-r-xgb-house-price/README.md))
Built static and interactive geospacial plots overlayed with feature data.
Feature engineered high/low important words that associate w/ price using NLP.
Trained/tuned/evaluated/deployed SageMaker Multi-class XGBoost on holdout data & submitted predicted binned price to Kaggle competition. Submission scored 0.8876 (mLogLoss), which would have placed 6th (out of 90 entries) in live competition.
Modified {SHAPforxgboost} package to generate multi-class SHapley Additive exPlanations (SHAP) values/plots that explain how XGBoost model made predictions. 

### Predicting Churn using AWS SageMaker & Local RStudio
* Utilized SageMaker and built-in XGBoost algorithm to train, tune, evaluate, and deploy model for predicting bank customer churn in the SLICED season 1 episode 7 Kaggle competition. ([Blog](https://toddwarczak.netlify.app/post/sagemaker/), [Github](https://github.com/TWarczak/data_warz/tree/master/content/post/2021-08-01-sagemaker-r-xgb-churn))
* Configured local RStudio to make API calls to SageMaker using SageMaker Python SDK and {reticulate} R package.
* Best performing model deployed as SageMaker endpoint for real-time predictions on holdout data (w/ minimal feature engineering and pre-processing). Predictions submitted to SLICED competition received a score of 0.07622 (LogLoss), which would have placed 8th out of 130 entries.

### Forecasting Daily Sales with Modeltime
* Forecasted 3 months of daily sales utilizing the {modeltime} package in R to help ‘Superstore’ company selling furniture, technology, and office supplies manage supply-chain and inventory decisions in Q1 (data source: Kaggle). ([Blog](https://toddwarczak.netlify.app/post/modeltime/), [Github](https://github.com/TWarczak/data_warz/tree/master/content/post/2021-05-26-forecasting-daily-sales-w-modeltime))
* Exploratory data analysis and 11 models tested. 6 {tidymodels} workflows tuned for individual forecasts and weighted ensemble (Support Vector Machine/Neural Network-AR/Random Forest/Prophet-XGboost) forecast.

### TidyTuesday
* Weekly twitter project focusing on cleaning, wrangling, summarizing, and arranging a new dataset in R to produce a single chart. Typically using {ggplot2} and {tidyverse} tools. Shared via #TidyTuesday. [github-TidyTuesday](https://github.com/TWarczak/TidyTuesday)
* Mario Kart 64 World Records - Visualization of cumulative days individual records are held by players.
* Survivor - Visualization showing Myers-Briggs Type Indicator personalities of all show contestants vs. the winners.

## Data Science Skills {.divider}
  * R ★★★★
  * Python ★★
  * AWS SageMaker/S3/EC2...
  * SQL
  * GitHub
  * Markdown/Jupyter Notebooks
  * Data Cleaning/Wrangling
  * Data Visualization
  * Probability/Statistics
  * Machine Learning
  * Regression/Classification
  * Time-Series Forecasting
  * TensorFlow/Tidymodels/H2O...
