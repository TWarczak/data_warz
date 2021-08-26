In this post I explore an Austin Housing dataset and predict binned housing price. EDA includes static and interactive geospacial feature maps and feature engineering using natural language processing (NLP). After training/tuning multi-class XGBoost models , I run batch inference to predict the price of Austin, TX houses. I then submit predictions to the Kaggle competition which scrored 0.8876 (mlogloss), which would have placed 6th in the live competition. After submission, I generate SHapley Additive exPlanations (SHAP) plots to understand how XGBoost made predictions. [data warz (blog post)](https://toddwarczak.netlify.app/post/xgb-multi-class/)

*** 

### [SageMaker + RStudio to Predict Home Prices w/ Multi-class XGBoost; Explaining Model Behavior with Geospacial Plots and SHAP](https://github.com/TWarczak/data_warz/tree/master/content/post/2021-08-01-austin-r-xgb-house-price) Source: [Kaggle](https://www.kaggle.com/c/sliced-s01e11-semifinals)

***

![.content/post/2021-08-01-austin-r-xgb-house-price/featured-austin.png](https://raw.githubusercontent.com/TWarczak/data_warz/master/content/post/2021-08-01-austin-r-xgb-house-price/featured-austin.png)

***

![.content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-3-1.png](https://raw.githubusercontent.com/TWarczak/data_warz/master/content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-3-1.png)

***

![.content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-10-1.png](https://raw.githubusercontent.com/TWarczak/data_warz/master/content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-10-1.png)

***

![.content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-35-1.png](https://raw.githubusercontent.com/TWarczak/data_warz/master/content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-35-1.png)

***

![.content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-42-1.png](https://raw.githubusercontent.com/TWarczak/data_warz/master/content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-42-1.png)

***

![.content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-80-1.png](https://raw.githubusercontent.com/TWarczak/data_warz/master/content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-80-1.png)

***

![.content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-79-1.png](https://raw.githubusercontent.com/TWarczak/data_warz/master/content/post/2021-08-01-austin-r-xgb-house-price/index_files/figure-markdown_strict/index-79-1.png)

***
