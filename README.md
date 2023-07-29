
# Kannada MNIST Classification

This project aims to classify handwritten digits from the Kannada MNIST dataset using various machine learning classifiers. The dataset consists of 28x28 grayscale images of handwritten digits from 0 to 9, written in the Kannada script.

## Dataset
The Kannada MNIST dataset is loaded from .npz files containing training and testing data. The images are reshaped into 1D arrays, and PCA is applied to reduce the dimensionality to 10 components.
## Classifiers
The following classifiers are trained and evaluated on the reduced dataset:

    Decision Tree
    Random Forest
    Gaussian Naive Bayes
    K-Nearest Neighbors (KNN)
    Support Vector Machine (SVM)
## Evaluation Metrics

For each classifier, the following evaluation metrics are calculated:

    Precision
    Recall
    F1 Score
    Confusion Matrix
    ROC-AUC Score (one-vs-rest strategy)
## Usage

1) Make sure you have the necessary dependencies installed by running: pip install numpy pandas seaborn matplotlib scikit-learn yellowbrick

2) Download the Kannada MNIST dataset and save the .npz files in the DataSet directory.

3) Run the main.py script to load the dataset, perform PCA, train the classifiers, and evaluate their performance. The evaluation metrics will be printed for each classifier.

4) The ROC-AUC curves for each classifier can also be visualized using the Yellowbrick library.
## Acknowledgments

1) The Kannada MNIST dataset is available on Kaggle: Kannada MNIST
2) The project uses the scikit-learn and Yellowbrick libraries for machine learning and visualization
## License

MIT License

Copyright (c) 2023 hariprasath-jayakumar
