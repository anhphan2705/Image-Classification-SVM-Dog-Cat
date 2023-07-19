import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn.svm import SVC
import numpy as np
import os

## Define file directories
file_dir = './data-shorten'
output_dir = './output/VGG16_trained.pth'
TRAIN = 'train' 
VAL = 'val'
TEST = 'test'

def get_data(file_dir):
    """
    Load and transform the data using PyTorch's ImageFolder and DataLoader.

    Args:
        file_dir (str): Directory path containing the data.
        TRAIN (str, optional): Name of the training dataset directory. Defaults to 'train'.
        VAL (str, optional): Name of the validation dataset directory. Defaults to 'val'.
        TEST (str, optional): Name of the test dataset directory. Defaults to 'test'.

    Returns:
        datasets_img (dict): Dictionary containing the datasets for training, validation, and test.
        datasets_size (dict): Dictionary containing the sizes of the datasets.
        dataloaders (dict): Dictionary containing the data loaders for training, validation, and test.
        class_names (list): List of class names.
    """
    print("[INFO] Loading data...")
    # Initialize data transformations
    data_transform = {
        TRAIN: transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        VAL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        TEST: transforms.Compose([
            transforms.Resize(254),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }
    # Initialize datasets and apply transformations
    datasets_img = {
        file: datasets.ImageFolder(
            os.path.join(file_dir, file),
            transform=data_transform[file]
        )
        for file in [TRAIN, VAL, TEST]
    }
    # Load data into dataloaders
    dataloaders = {
        file: torch.utils.data.DataLoader(
            datasets_img[file],
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        for file in [TRAIN, VAL, TEST]
    }
    # Get class names and dataset sizes
    class_names = datasets_img[TRAIN].classes
    datasets_size = {file: len(datasets_img[file]) for file in [TRAIN, VAL, TEST]}
    for file in [TRAIN, VAL, TEST]:
        print(f"[INFO] Loaded {datasets_size[file]} images under {file}")
    print(f"Classes: {class_names}")

    return datasets_img, datasets_size, dataloaders, class_names


def get_vgg16_pretrained_model(weights=models.VGG16_BN_Weights.DEFAULT):
    """
    Retrieve the VGG-16 pre-trained model and remove the classifier layers.

    Args:
        model_dir (str, optional): Directory path for loading a pre-trained model state dictionary. Defaults to ''.
        weights (str or dict, optional): Pre-trained model weights. Defaults to models.vgg16_bn(pretrained=True).state_dict().
        len_target (int, optional): Number of output classes. Defaults to 1000.

    Returns:
        vgg16 (torchvision.models.vgg16): VGG-16 model with removed classifier.
    """
    print("[INFO] Getting VGG-16 pre-trained model...")
    # Load VGG-16 pretrained model
    vgg16 = models.vgg16_bn(weights)
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.requires_grad = False
    # Remove the classifier layers
    features = list(vgg16.classifier.children())[:-7]
    # Replace the model's classifier
    vgg16.classifier = nn.Sequential(*features)
    print(vgg16)
    return vgg16

def get_features(vgg):
    svm_features = []
    svm_labels = []
    
    for i, data in enumerate(dataloaders[TRAIN]):
        inputs, labels = data
        if use_gpu:
            # Get the data through the feature extractor of VGG16
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            # Extract data from VGG16 feature extractor as a vector
            features = vgg(inputs)
            features = features.cpu().detach().numpy().flatten()
            labels = labels.cpu().detach().numpy()
            # Add it to the features list
            svm_features.append(features)
            svm_labels.append(labels)
        else:
            # Get the data through the feature extractor of VGG16
            inputs = Variable(inputs)
            labels = Variable(labels)
            # Extract data from VGG16 feature extractor as a vector
            features = vgg(inputs)
            features = features.detach().numpy().flatten()
            labels = labels.cpu().detach().numpy()
            # Add it to the features list
            svm_features.append(features)
            svm_labels.append(labels)
        # features = np.ndarray(features)
        # features.flatten()
        
    return svm_features, svm_labels

def svm_classifier(features, labels):
    features = np.array(features)
    labels = np.array(labels)
    
    
    
    return None

if __name__ == '__main__':
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    # Get Data
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir)
    # Get VGG16 pre-trained model
    vgg16 = get_vgg16_pretrained_model()
    # vgg16 = get_vgg16_pretrained_model('./output/VGG16_trained.pth', len_target=2)      # If load custom pre-trained model, watch out to match len target
    # Move model to GPU
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    # Extract features and labels from the VGG16 model
    svm_features, svm_labels = get_features(vgg16)
