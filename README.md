# ECE 143 Project Mental Health & Social Media Analysis Notebook

This project implements a Mental Health & Social Media Analysis

## File Structure

* src/dataset.py:
  * this file defines the dataset class that load the dataframe from data/ as dataset
* src/model.py:
  * this file defines the statistical model for the analysis model.
  * this file will fit multiple model and select the best one for training dataset.
* src/plotter.py
  * this file defines plotting functions
* src/utils.py
  * this contains the utility function for the project

## Usage

To set up the environment, we need
`pip install -r requirements.txt`

* final.ipynb: it includes the whole pipeline: from data preprocessing to model fitting to visualization.
* analyzer_app.py: this one include a recommendation app. By running `python analyzer_app.py`, the user can input the data features and get the estimate of their stress level and hapiness index based on the inputs.

## Third-Party Module

* pandas
* numpy
* scikit-learn
* xgboost
* matplotlib
