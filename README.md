Gradient Boosting Regressor From Scratch

This project implements a Gradient Boosting Regression model from scratch using Decision Trees as base learners.

The project includes Decision Tree implementation from scratch using squared error split criteria, Gradient Boosting training using residual learning, and support for hyperparameters including number of estimators, learning rate, and tree depth.

The model was trained and evaluated on the Boston Housing dataset. Model performance was evaluated on a held-out test dataset.

The training curve shows steady loss reduction across boosting rounds, confirming correct gradient boosting behavior.

Project Deliverables include modular Python implementation, training and evaluation script, performance report with training curve, and a requirements file for reproducibility.

How to Run

Install dependencies:
pip install -r requirements.txt

Run training:
python train.py
