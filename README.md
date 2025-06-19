# Housing-Price-Prediction-Neural-Network-

Engineered a fully custom multi-layer perceptron (MLP) from scratch without using any machine learning libraries to classify housing listings by bedroom count using structured data from the New York real estate market (4,800+ samples, 16 features). The project emphasized algorithmic rigor, efficient computation, and architectural design within strict resource and runtime constraints.

- **Manual Neural Network Design:** Implemented a feedforward architecture with configurable depth, activation functions (ReLU, sigmoid), and backpropagation using only NumPy, including matrix-based gradient computation and weight updates.

- **Efficient Training Pipeline:** Vectorized all forward and backward passes to meet runtime limits (<5 min end-to-end) in a constrained environment, using techniques such as mini-batch gradient descent and early stopping.

- **Feature Engineering:** Performed domain-aware preprocessing including one-hot encoding for high-cardinality categorical variables and normalization for numeric attributes; handled text fields with customized embeddings and reduced dimensionality.

- **Hyperparameter Optimization:** Conducted structured tuning across learning rates, batch sizes, layer widths, and initialization schemes; recorded validation accuracy across 5 train-test splits to empirically guide model selection.

- **Performance Benchmarking:** Consistently achieved >90% of baseline model accuracy across randomized evaluations, demonstrating robustness and generalizability of the custom-built network.
