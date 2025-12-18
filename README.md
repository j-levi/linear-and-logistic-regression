# Linear & Logistic Regression

## Overview
This folder collects experiments that compare two workhorses of supervised learning: linear regression for continuous targets and logistic regression for binary labels. The focus is on intuition, behavior, and trade-offs rather than derivations.

## Linear regression (fit a line/plane)
- Predicts a numeric outcome by combining input features with learned weights.
- Ridge regression adds a penalty on large weights to reduce overfitting and stabilize solutions.
- Useful when relationships are roughly linear and you want interpretable coefficients.

## Logistic regression (fit a boundary)
- Predicts the probability of belonging to a class using the logistic (sigmoid) link.
- Optimized with gradient-based methods; regularization (L1/L2) keeps models compact and well-behaved.
- Works well as a strong baseline classifier and is easy to calibrate and explain.

## Practical notes
- Scale or standardize features before training to help optimization.
- Use validation curves to pick regularization strength; watch for class imbalance.
- Compare full-batch solvers to mini-batch/SGD when datasets get large.

## Files
- `Project1JL.ipynb`: experiments with linear and logistic regression fits, feature scaling, and regularization.
- `Assignment 1`: supporting materials and exercises.
