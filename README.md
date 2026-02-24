Custom RBF Layer Implementation with Keras/TensorFlow

This repository contains a professional implementation of a **Radial Basis Function (RBF)** neural network using a custom Keras layer. The model is trained and tested on the classic **Iris Dataset**.

## 🚀 Features
* **Custom Keras Layer:** Inherits from `tf.keras.layers.Layer` to implement RBF logic.
* **Mathematical Precision:** Computes Euclidean distance between inputs and trainable centers
* **High Performance:** Achieves 98% accuracy on the Iris classification task.
* **Visualization:** Includes training history plots for accuracy and loss.

## 🧠 Mathematical Background
The RBF layer uses the following Gaussian activation function:

$$RBF(x) = \exp(-\gamma \cdot \|x - \mu\|^2)$$

Where:
* $\mu$ represents the trainable centers.
* $\gamma$ (gamma) is the spread parameter.
* $\|x - \mu\|^2$ is the squared Euclidean distance.
