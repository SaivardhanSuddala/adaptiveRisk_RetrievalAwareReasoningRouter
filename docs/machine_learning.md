# Machine Learning Reference Notes

## Gradient Descent
Gradient descent updates parameters in the direction opposite the gradient of
the loss function: `theta = theta - learning_rate * gradient`. A learning rate
that is too large can cause the loss to diverge instead of converge; a rate
that is too small converges slowly. Stochastic gradient descent computes the
gradient on a single example or mini-batch rather than the full dataset,
trading gradient accuracy for update speed.

## Overfitting and Regularization
A model overfits when it fits noise in the training data rather than the
underlying pattern, showing low training error but high validation error.
L2 regularization (weight decay) penalizes large weights by adding a term
proportional to the squared weight magnitude to the loss function. Dropout
randomly zeroes activations during training to prevent co-adaptation of units.

## Bias-Variance Tradeoff
High-bias models (e.g. linear models on nonlinear data) underfit and have high
training error. High-variance models (e.g. deep trees with no pruning) overfit
and generalize poorly. Total expected error decomposes into bias squared,
variance, and irreducible noise.

## Evaluation Metrics for Classification
Precision is true positives divided by predicted positives. Recall is true
positives divided by actual positives. F1 score is the harmonic mean of
precision and recall, useful when classes are imbalanced and accuracy alone is
misleading.

## Retrieval-Augmented Generation
RAG combines a retriever, which returns relevant passages from an external
index, with a generator, which conditions its output on those passages. This
grounds generated text in retrieved evidence rather than relying solely on
parameters learned during pretraining.
