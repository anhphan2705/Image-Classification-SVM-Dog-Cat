import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import pickle

## Define file directories
file_dir = "./data-shorten"
output_dir = "./output/VGG16_SVM_trained.pth"
TRAIN = "train"
VAL = "val"
TEST = "test"


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
        TRAIN: transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        VAL: transforms.Compose(
            [
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor()
            ]
        ),
        TEST: transforms.Compose(
            [
                transforms.Resize(254), 
                transforms.CenterCrop(224), 
                transforms.ToTensor()
            ]
        ),
    }
    # Initialize datasets and apply transformations
    datasets_img = {
        file: datasets.ImageFolder(
            os.path.join(file_dir, file), transform=data_transform[file]
        )
        for file in [TRAIN, VAL, TEST]
    }
    # Load data into dataloaders
    dataloaders = {
        file: torch.utils.data.DataLoader(
            datasets_img[file], batch_size=8, shuffle=True, num_workers=4
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


def get_vgg16_modified_model(weights=models.VGG16_BN_Weights.DEFAULT):
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
    # print(vgg16)
    return vgg16


def get_features(vgg, file=TRAIN):
    print(f"[INFO] Getting '{file}' features...")
    svm_features = []
    svm_labels = []
    data_batches_len = len(dataloaders[file])
    for i, data_batch in enumerate(dataloaders[file]):
        print(f"\r[FEATURE] Loading batch {i + 1}/{data_batches_len} ({len(data_batch[1])*(i+1)} images)", end='', flush=True)
        # In this case, loaded databatch of 8 images including 8 features and 8 labels
        inputs, labels = data_batch
        if use_gpu:
            # Get the data through the feature extractor of VGG16
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            # Extract data from VGG16 feature extractor as a vector
            features = vgg(inputs)
            # print(features.shape)     # torch.Size([8, 25088])
            # print(labels.shape)       # torch.Size([8])
            features = features.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        else:
            # Get the data through the feature extractor of VGG16
            inputs = Variable(inputs)
            labels = Variable(labels)
            # Extract data from VGG16 feature extractor as a vector
            features = vgg(inputs)
            features = features.detach().numpy()
            labels = labels.detach().numpy()
        
        # Add feature with correct label into an array
        # print(features.shape)     # (8, 25088)
        # print(labels.shape)       # (8,)
        for index in range(len(labels)):
            feature = features[index]  
            label = labels[index]
            # Add it to the features list
            # print(feature.shape)     # (25088,)
            svm_features.append(feature)
            # print(label.shape)       # (1)
            svm_labels.append(label)
            
    print("\n[FEATURE] Features loaded")
    return svm_features, svm_labels


def svm_classifier(train_data, test_data, epochs=5):
    since = time.time()
    FEATURE_INDEX = 0
    LABEL_INDEX = 1
    # There are 1000 images in the train data
    train_features = np.array(train_data[FEATURE_INDEX])
    # print(features.shape)     # (1000, 25088)
    train_labels = np.array(train_data[LABEL_INDEX])
    # print(labels.shape)       # (1000,)
    
    # There are 600 images in the test data
    test_features = np.array(test_data[FEATURE_INDEX])
    # print(features.shape)     # (1000, 25088)
    test_labels = np.array(test_data[LABEL_INDEX])
    # print(labels.shape)       # (1000,)
    scores = []
    svm_model = SVC(gamma="auto")
    for epoch in range(epochs):
        print(f"[SVM Model] Training epoch {epoch+1}/{epochs}...")
        svm_model.fit(train_features, train_labels)
        score = svm_model.score(test_features, test_labels)
        print(f"[SVM Model] The score for epoch {epoch+1} is: {score:.4f}\n")
        scores.append(score)
    elapsed_time = time.time() - since
    print(f"[INFO] Model has score mean: {np.mean(scores):.4f} and standard deviation: {np.std(scores):.4f} in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    return svm_model, np.mean(scores), np.std(scores)


if __name__ == "__main__":
    # Use GPU if available. Note that this only to load features using VGG16. Scikit Learn SVM does not support GPU
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    # Get Data
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir)
    # Get VGG16 pre-trained model
    vgg16 = get_vgg16_modified_model()
    # Move model to GPU
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    # Extract features and labels from the VGG16 model
    svm_train_features, svm_train_labels = get_features(vgg16, TRAIN)
    svm_test_features, svm_test_labels = get_features(vgg16, TEST)
    # Run SVM 
    svm_model, score_mean, score_stdv = svm_classifier(
        [svm_train_features, svm_train_labels],
        [svm_test_features, svm_test_labels],
        epochs=5
    )
    # Save model
    print('[INFO] Saving model...')
    pickle.dump(svm_model, open(output_dir, 'wb'))
    print('[INFO] Done')
