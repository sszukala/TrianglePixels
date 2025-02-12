# TrianglePixels
The results obtained from the experiments are compared and analyzed to determine the effectiveness of using triangle pixels in image classification tasks. 

# Triangle Pixel Model for Image Classification

## Overview

This repository explores the concept of utilizing triangle pixels instead of traditional square pixels for image classification tasks. The primary goal is to investigate the potential benefits of this alternative pixel structure in terms of memory efficiency and visual perception capabilities. The project uses PyTorch to implement and train convolutional neural networks (CNNs) with both square and triangle pixel representations.

## Key Features

* **Triangle Pixel Representation:** Introduces a custom layer (`TrianglePixelSplitter`) to split square pixels into two triangular pixels, effectively doubling the number of channels in the input data.
* **Teacher-Student Architecture:** Employs a knowledge distillation technique, where a larger, more complex teacher model (square pixel-based) guides the learning of a smaller, more efficient student model (triangle pixel-based).
* **Memory Efficiency:** Analyzes the memory usage of both models to compare the storage requirements of triangle and square pixel representations.
* **Visual Perception:** Evaluates the performance of both models on FashionMNIST and CIFAR-10 datasets to assess the impact of pixel structure on visual perception tasks.

## Methodology

### Data Preprocessing:

* **FashionMNIST:** Images are transformed to tensors, followed by Gaussian blur, median blur, and histogram equalization to enhance image features and robustness.
* **CIFAR-10:** Images are transformed to tensors and normalized to a specific range for consistency and better training performance.

### Model Training:

* **Teacher Model (Square Pixel):** A CNN with two convolutional layers, batch normalization, max pooling, and two fully connected layers is used as the teacher model.
* **Student Model (Triangle Pixel):** A smaller CNN architecture with one convolutional layer, batch normalization, max pooling, and two fully connected layers is used as the student model.
* **Knowledge Distillation:** The student model is trained using knowledge distillation, where it learns from the teacher model's predictions (logits) rather than just ground truth labels.
* **Early Stopping:** Implemented to prevent overfitting and to achieve optimal model performance.

### Evaluation Metrics:

* **Accuracy:** Measures the percentage of correctly classified images.
* **Precision:** Indicates the proportion of correctly predicted positive instances.
* **Recall:** Represents the proportion of actual positive instances correctly identified.
* **F1 Score:** Provides a harmonic mean of precision and recall, offering a balanced performance evaluation.

## Results and Analysis

The results obtained from the experiments are compared and analyzed to determine the effectiveness of using triangle pixels in image classification tasks. The key findings are summarized as follows:

* **Memory Efficiency:** The student model (triangle pixel) demonstrates significantly reduced memory footprint compared to the teacher model (square pixel) due to its smaller architecture.
* **Visual Perception:** The student model's performance on the FashionMNIST and CIFAR-10 datasets, though slightly lower in terms of accuracy, indicates the potential of triangle pixels in learning discriminative features.
* **Batch Processing:** Triangle Pixel Model shows faster batch processing time in comparison to Square Pixel Model.

## Conclusion and Future Work

The results highlight the trade-offs between memory efficiency, visual perception, and computational performance when using triangle pixels instead of traditional square pixels in CNNs. Future work might involve exploring more complex image datasets, optimizing the triangle pixel model architecture, and investigating potential benefits for real-world applications such as edge devices or embedded systems where memory constraints are a significant consideration.

## Getting Started

1. Clone the repository: `git clone https://github.com/your-username/triangle-pixel-model.git`
2. Install necessary packages: `pip install torchvision tqdm scipy scikit-learn psutil`
3. Run the notebook: `jupyter notebook`

## Contribution Guidelines

If you'd like to contribute to this project, please feel free to fork the repository and submit pull requests. Any improvements or new ideas are welcome!

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, please contact [sszukala@gmail.com].
