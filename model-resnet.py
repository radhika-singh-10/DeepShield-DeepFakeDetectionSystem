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

class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Use ResNet-50 as the backbone
        # Replace the final fully connected layer with a binary classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.resnet(x)
    
def main():
    transform = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
    model = ResNetBinaryClassifier(num_classes=num_classes)
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

        torch.save(model.state_dict(), 'resnet_model.pth')
        metrics = {
            "training_losses": training_losses,
            "training_accuracies": training_accuracies,
            "validation_losses": validation_losses,
            "validation_accuracies": validation_accuracies,
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores
        }
        with open("resnet_model.pkl", "wb") as f:
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
        print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        return val_loss, val_accuracy, precision, recall, f1

    metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100)
    print(metrics)


if __name__ == "__main__":
    main()

# # Initialize the model
# model = ResNetBinaryClassifier()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Load datasets
# train_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/train-base', transform=transform)
# val_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/val-base', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# # Define loss, optimizer, and scheduler
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# # Training and validation functions (unchanged from the original code)
# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     training_losses, training_accuracies = [], []
#     validation_losses, validation_accuracies = [], []
#     precisions, recalls, f1_scores = [], [], []
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct, total = 0, 0
        
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         scheduler.step()
#         training_loss = running_loss / len(train_loader)
#         training_accuracy = 100 * correct / total
#         training_losses.append(training_loss)
#         training_accuracies.append(training_accuracy)
        
#         print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%")
#         val_loss, val_accuracy, precision, recall, f1 = validate_model(model, val_loader, criterion, device)
#         validation_losses.append(val_loss)
#         validation_accuracies.append(val_accuracy)
#         precisions.append(precision)
#         recalls.append(recall)
#         f1_scores.append(f1)
#     return training_losses, training_accuracies, validation_losses, validation_accuracies, precisions, recalls, f1_scores

# def validate_model(model, val_loader, criterion, device):
#     model.eval()
#     correct, total = 0, 0
#     running_loss = 0.0
#     all_labels = []
#     all_predictions = []
    
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             all_labels.extend(labels.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())
    
#     val_loss = running_loss / len(val_loader)
#     val_accuracy = 100 * correct / total
#     precision = precision_score(all_labels, all_predictions, average='weighted')
#     recall = recall_score(all_labels, all_predictions, average='weighted')
#     f1 = f1_score(all_labels, all_predictions, average='weighted')
    
#     print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
#     print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
#     return val_loss, val_accuracy, precision, recall, f1

# # Train and evaluate the model
# metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)
# print(metrics)
