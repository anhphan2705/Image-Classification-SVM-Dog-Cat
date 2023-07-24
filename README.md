# Image Classification for Dogs and Cat using VGG-16 and SVM

This repository contains a Python script for image classification using a pre-trained VGG-16 model and an SVM (Support Vector Machine) classifier. The script uses PyTorch for data handling and feature extraction, and scikit-learn for training and evaluating the SVM classifier.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Image classification is a fundamental task in computer vision, where the goal is to categorize images into different classes or categories. This repository presents a solution to the image classification problem using a two-step process:

1. Feature Extraction: We use a pre-trained VGG-16 model, a popular deep learning architecture, to extract high-level features from the images. The last few layers of VGG-16 (the classifier) are removed, and the output of the remaining layers is used as the feature representation for each image.

2. SVM Classification: The extracted features are then used to train an SVM classifier, which learns to distinguish between different image classes based on the feature representations.

## Requirements

To run the script, you need the following dependencies:

- Python (>=3.6)
- PyTorch (>=1.0.0)
- torchvision
- scikit-learn
- numpy
- matplotlib (optional, for visualizing results)

You can install the required packages using the following command:

```bash
pip install torch torchvision scikit-learn numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/image-classification.git
cd image-classification
```

2. Download pre-trained model

- I have already had a model fitted with accuracy of 94.71%. You can download it here

    ```
    https://www.dropbox.com/s/mkcj9486umgnim8/SVM_Train_9471.zip?dl=0
    ```

3. Prepare your image dataset in the following directory structure:

```
cat_dog_SVM_classifier.py
SVM_trained_9471.pth (optional)
data-shorten/
    train/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
    test/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
```

4. Modify the script parameters in the `main.py` file:

- `file_dir`: Path to the root directory containing the dataset.
- `output_dir`: Path to save the trained SVM model.
- You can adjust batch size and other hyperparameters as needed.

5. Run the script:

```bash
python main.py
```

## Data Preparation

Make sure your dataset is organized into separate train and test directories, each containing subdirectories for different classes. The script uses PyTorch's `ImageFolder` class to automatically load and transform the images for training and testing.

## Model Training and Evaluation

The script uses a pre-trained VGG-16 model with its classifier removed to extract features from the images. These features are then used to train an SVM classifier using scikit-learn's `SVC` class. The trained SVM model is evaluated on the test dataset, and the accuracy score is reported.

## Results

Upon running the script, you will see the training progress, and the final accuracy score of the trained SVM classifier on the test dataset will be displayed. The trained SVM model will be saved in the specified output directory as a pickled file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The VGG-16 model used in this project is part of the torchvision library.
- The code is inspired by various tutorials and online resources, and their links are provided in the comments within the script.