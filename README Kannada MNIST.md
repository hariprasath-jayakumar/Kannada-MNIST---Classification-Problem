
## Title-Kannada MNIST-Classification Problem

Kannada MNIST Recognition is a machine learning project that aims to recognize handwritten digits from the Kannada MNIST dataset. The dataset contains images of handwritten digits in the Kannada script, similar to the English MNIST dataset.

This project uses various machine learning algorithms to train models capable of accurately classifying the digits. 
## Dataset
The Kannada MNIST dataset used for this project can be obtained from the following source:
URL : "https://www.kaggle.com/datasets/higgstachyon/kannada-mnist"
## Requirements

Python

NumPy

Scikit-learn

## Procedures 

1) Extract Dataset from the Given URL
2) Align the data for Train & Test
3) Reshape the data to 28*28
4) Perform PCA to componenets
5) Apply the below classifier

    • Decision Trees

    • Random forest

    • Naive Bayes Model

    • K-NN Classifier

    • SVM
6) For each of this method produce the following metrics:
• Precision

• Recall

• F1 - Score

• Confusion Matrix

• RoC - AUC curve

7) Try to repeat the same experiment for different component size : 15,20,25,30
## License
This project is licensed under the MIT License.