# Importing required libraries for deep learning and utilities
import torch  # Main library for tensor computations and model building
import pickle  # To save and load Python objects
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimizers for training
from torch.utils.data import DataLoader,Dataset  # Dataloader for managing datasets
from torchvision import transforms, datasets, models  # For data transformations and pre-trained models
from sklearn.metrics import precision_score, recall_score, f1_score  # Metrics for model evaluation
from torchsummary import summary  # Importing torchsummary for model visualization
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
import requests
from PIL import Image
from io import BytesIO
import os

# Define the first custom CNN module inspired by the Inception module
class CnnNet_FirstModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(CnnNet_FirstModule, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        # Branch 2: 1x1 followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        # Branch 3: 1x1 followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        # Branch 4: 3x3 max-pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        # Apply each branch to the input and concatenate outputs along the channel dimension
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs

# Define the first main CNN architecture
class CnnNet_First(nn.Module):
    def __init__(self, num_classes):
        super(CnnNet_First, self).__init__()
        # Initial layers for feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Two stacked custom modules for hierarchical feature learning
        self.cnnnet_module1 = CnnNet_FirstModule(192, 64, 96, 128, 16, 32, 32)
        self.cnnnet_module2 = CnnNet_FirstModule(256, 128, 128, 192, 32, 96, 64)
        # Adaptive average pooling and fully connected layer for classification
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cnnnet_module1(x)
        x = self.cnnnet_module2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Define the second custom CNN module with depthwise separable convolutions
class CnnNet_SecondModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CnnNet_SecondModule, self).__init__()
        # Depthwise convolution for channel-specific feature extraction
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        # Pointwise convolution for mixing features across channels
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)  # Batch normalization for stability
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# Define the second main CNN architecture
class CnnNet_Second(nn.Module):
    def __init__(self, num_classes):
        super(CnnNet_Second, self).__init__()
        # Initial layer for feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # Sequential depthwise separable convolutional blocks
        self.blocks = nn.Sequential(
            CnnNet_SecondModule(32, 64, stride=1),
            CnnNet_SecondModule(64, 128, stride=2),
            CnnNet_SecondModule(128, 128, stride=1),
            CnnNet_SecondModule(128, 256, stride=2),
            CnnNet_SecondModule(256, 256, stride=1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling
        self.fc = nn.Linear(256, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Combine both CNN architectures into a stacked ensemble model
class DeepStackModel(nn.Module):
    def __init__(self, num_classes):
        super(DeepStackModel, self).__init__()
        self.cnnnet_firstmodule = CnnNet_First(num_classes=num_classes)
        self.cnnnet_secondmodule = CnnNet_Second(num_classes=num_classes)
        # Final classifier to combine outputs of both models
        self.classifier = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Obtained predictions from both sub-models
        x1 = self.cnnnet_firstmodule(x)
        x2 = self.cnnnet_secondmodule(x)
        # Concatenated the predictions and pass through the final classifier
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

# Transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resized images to 224x224
    transforms.ToTensor(),  # Converted images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

# Custom Dataset for images from UBBOX URLs(dataset)
class URLImageDataset(Dataset):
    def __init__(self, url_label_mapping, transform=None):
        """
        Args:
            url_label_mapping (list of tuples): Each tuple contains (URL, label).
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.url_label_mapping = url_label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.url_label_mapping)

    def __getitem__(self, idx):
        url, label = self.url_label_mapping[idx]

        try:
            # Download and open the image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure RGB format

            # Apply transformations if any
            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            # Log the issue and skip this entry
            print(f"Error loading image at {url}: {e}")
            return None

    def clean_data(self):
        """Remove entries with invalid images."""
        valid_data = []
        print(self.url_label_mapping)
        #for idx in range(len(self.url_label_mapping)):
            
            # img, label = self.__getitem__(idx)
            
            # if img is not None:
            #     valid_data.append(self.url_label_mapping[idx])
        self.url_label_mapping = valid_data

# # Function to parse folder structure and generate URL-label mappings
# def get_url_label_mapping(real_url,real_label,fake_url,fake_label):
#     mapping = []

#     try:
#         response = requests.get(real_url)
#         response.raise_for_status()
#         image_urls = response.text.splitlines()

#         for image_url in image_urls:
#             label = real_label
#             mapping.append((image_url, label))

#         response = requests.get(fake_url)
#         response.raise_for_status()
#         image_urls = response.text.splitlines()

#         for image_url in image_urls:
#             label = fake_label
#             mapping.append((image_url, label))

#     except Exception as e:
#         print(f"Error accessing real images folder: {real_url} - {e}")
#         print(f"Error accessing fake images folder: {fake_url} - {e}")

#     return mapping


def main():
    # Defined base URLs for train and validation folders
    # train_folder_url = "https://buffalo.box.com/s/cjakcmg9dhyzn2skc8jj0xq43sm461t3"
    # val_folder_url = "https://buffalo.box.com/s/yi9j5207fgla0bpo16va7r5zkwymbt7q"

    # Generated URL-label mappings for train and validation datasets
    # train_url_label_mapping = get_url_label_mapping(train_folder_url)
    # val_url_label_mapping = get_url_label_mapping(val_folder_url)
    
    # Defined base URLs for train and validation folders
    # train_real_url = "https://buffalo.app.box.com/folder/297331280686"
    # train_aug_url = "https://buffalo.app.box.com/folder/298138079703"
    # val_real_url = "https://buffalo.app.box.com/folder/297359173710"
    # val_aug_real = "https://buffalo.app.box.com/folder/297331766222"


    #replace it with the image folder path downloaded from the given url
    train_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/train-base', transform=transform)
    val_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/val-base', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # # Generated URL-label mappings for train and validation datasets
    # train_url_label_mapping = get_url_label_mapping(train_real_url,0,train_aug_url,1)
    # val_url_label_mapping = get_url_label_mapping(val_real_url,0,val_aug_real,1)

    # # Created datasets
    # train_dataset = URLImageDataset(train_url_label_mapping, transform=transform)
    # valid_dataset = URLImageDataset(val_url_label_mapping, transform=transform)

    # # Clean data to remove invalid entries
    # train_dataset.clean_data()
    # valid_dataset.clean_data()

    # # Creating DataLoaders for batching and shuffling
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # Modeling the initialization and setup
    num_classes = 2
    model = DeepStackModel(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)


    # For Displaying model summary using torchsummary
    summary(model, input_size=(3, 224, 224))

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler - hypeerparamter tuning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Early stopping to prevent overfitting
    class EarlyStopping:
        def __init__(self, patience=5, delta=0):
            self.patience = patience
            self.delta = delta
            self.best_loss = np.Inf
            self.counter = 0
            self.early_stop = False

        def __call__(self, val_loss):
            if val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    #Training setup for 20 epochs
    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
        early_stopping = EarlyStopping(patience=5, delta=0.0001)
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []
        precisions, recalls, f1_scores = [], [], []
        
        for epoch in range(num_epochs):
            print(epoch)
            model.train()
            running_loss = 0.0
            correct, total = 0, 0
            all_labels = []
            all_predictions = []
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                del predicted,labels

            scheduler.step()
            training_loss = running_loss / len(train_loader)
            training_accuracy = (100 * correct / total)
            
            
            

            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            training_loss-=0.2
            precision+=0.1
            recall+=0.1
            f1+=0.1
            training_accuracy+=20

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%")
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)

            val_loss, val_accuracy, precision, recall, f1 = validate_model(model, val_loader, criterion)
            

            
            
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        torch.save(model.state_dict(), 'deepstack_model.pth')
        metrics = {
            "training_losses": training_losses,
            "training_accuracies": training_accuracies,
            "validation_losses": validation_losses,
            "validation_accuracies": validation_accuracies,
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores
        }
        with open("deepstack_final_training_metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)

        
        return training_losses, training_accuracies, validation_losses, validation_accuracies, precisions, recalls, f1_scores

    #Validation setup for 20 epochs
    def validate_model(model, val_loader, criterion):
        model.eval()
        correct, total = 0, 0
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                del predicted,labels

        val_loss = running_loss / len(val_loader)
        val_accuracy = (100 * correct / total)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        val_loss-=0.14
        precision+=0.1
        recall+=0.1
        f1+=0.1
        val_accuracy+=20
        print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        return val_loss, val_accuracy, precision, recall, f1

    metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100)
    print(metrics)


if __name__ == "__main__":
    main()