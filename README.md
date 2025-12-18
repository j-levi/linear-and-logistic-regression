# Linear & Logistic Regression (Mathematical Notes)

## Overview
This project contains implementations and experiments for linear regression (ordinary least squares, ridge) and logistic regression for binary classification. The README outlines the objective functions, closed-form solutions, and optimization methods.

## Linear Regression (OLS)
Given design matrix X ∈ R^{n×d} and targets y ∈ R^n, Ordinary Least Squares minimizes:
$$
\min_w \ \|X w - y\|^2_2.
$$
Normal equations give the closed-form solution (if X^T X is invertible):
$$
w^* = (X^T X)^{-1} X^T y.
$$
Ridge regression (L2 regularization) adds a penalty λ>0:
$$
\min_w \ \|X w - y\|^2_2 + \lambda \|w\|^2_2 \Rightarrow w^* = (X^T X + \lambda I)^{-1} X^T y.
$$

## Logistic Regression
For binary labels y_i ∈ {0,1}, model the probability with the logistic (sigmoid) function:
$$
p(y=1|x;w) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}.
$$
The negative log-likelihood (cross-entropy loss) is:
$$
L(w) = -\sum_{i=1}^n \left[ y_i \log \sigma(w^T x_i) + (1-y_i) \log (1-\sigma(w^T x_i)) \right].
$$
Gradient of the loss:
$$
\nabla_w L = X^T (\sigma(Xw) - y).
$$
Optimization is typically done with gradient descent, Newton-Raphson (IRLS), or quasi-Newton methods (L-BFGS). Regularization terms (L2, L1) can be added to the objective.

## Optimization Notes
- Feature scaling improves conditioning.
- For large n, stochastic gradient descent (SGD) or mini-batch methods are effective.
- Newton's method uses Hessian H = X^T W X (for logistic, W is diagonal with σ(1-σ)).

## Evaluation Metrics
- Regression: MSE, R^2.
- Classification: accuracy, precision/recall, ROC AUC, log-loss.

## References
- Bishop, Pattern Recognition and Machine Learning.

## Files
See Project1JL.ipynb and Assignment 1 for notebooks and exercises.
