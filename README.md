 
# Machine Learning Module Final Project

 

## Objective
The primary goal of this project is to implement a graphical application that simplifies the use of Machine Learning algorithms for all users, ranging from beginners to experts. This application should be intuitive and user-friendly, making advanced machine learning techniques accessible to a broad audience.

## Application Features:
1. **Upload a Dataset  :**
   - Provide functionality for users to upload their datasets in various formats.

2. **Display Dataset Statistics (Dashboard) :**
   - Create a dashboard to show key statistics and visualizations for the uploaded dataset.

3. **Data Preprocessing  :**
   - Implement features to preprocess the data, including handling missing values, data normalization, and more.

4. **Apply Various Machine Learning Algorithms  :**
   - Enable the application of different machine learning algorithms to the dataset.

5. **Display Algorithm Performance (Accuracy) :**
   - Show the performance metrics (accuracy) for each machine learning algorithm used.

6. **Adjust Algorithm Parameters to Improve Performance (Accuracy)  :**
   - Provide options to tweak algorithm parameters and improve their performance metrics.

7. **Application Design/Implementation :**
   - Focus on the overall design and user interface of the application, ensuring it is both aesthetically pleasing and functional.

## Tools and Technologies:

### Django Modules
- **File Uploading and Management:**
  ```python
  from django.shortcuts import render, redirect, get_object_or_404
  from django.conf import settings
  from .forms import UploadFileForm
  from .models import Dataset, UploadedFile
  from django.http import JsonResponse
  ```

- **Data Handling and Processing:**
  ```python
  import pandas as pd
  import dask.dataframe as dd
  import os
  import openpyxl
  import csv
  from io import BytesIO
  import base64
  import itertools
  import logging
  import numpy as np
  ```

- **Visualization:**
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  ```

- **Machine Learning Libraries:**
  ```python
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.impute import SimpleImputer
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.decomposition import PCA
  from sklearn.model_selection import train_test_split
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve

  from sklearn.ensemble import (
      RandomForestClassifier, RandomForestRegressor, 
      GradientBoostingClassifier, GradientBoostingRegressor,
      AdaBoostClassifier, AdaBoostRegressor,
  )
  from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
  from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
  from sklearn.svm import SVC, SVR
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import GaussianNB
  import xgboost as xgb
  import lightgbm as lgb
  ```

### Additional Information
- **Framework:** The application is developed using the Django framework, a high-level Python Web framework that encourages rapid development and clean, pragmatic design. Django is well-suited for creating robust web applications quickly.
  
- **Machine Learning:** The application leverages popular Python libraries such as `scikit-learn`, `xgboost`, and `lightgbm` for implementing machine learning algorithms and performing data processing tasks. These libraries are industry standards for machine learning and data analysis.

- **Data Visualization:** Libraries such as `matplotlib` and `seaborn` are used to create comprehensive and insightful data visualizations, aiding in the interpretation of dataset statistics and algorithm performance.

### Getting Started
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AbdellatifTouriri/ML_analitics.git
   cd ML_analitics
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python manage.py runserver
   ```

4. **Access the Application:**
   Open your web browser and navigate to `http://127.0.0.1:8000`.
 
