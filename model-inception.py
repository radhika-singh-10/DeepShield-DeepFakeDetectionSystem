import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from PIL import Image
import os
import pandas as pd
from torchsummary import summary 
import pickle

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datasets and DataLoaders
train_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/train-base', transform=transform)
val_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/val-base', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model setup
inception = models.inception_v3(pretrained=True)
inception.fc = nn.Linear(inception.fc.in_features, 2)
inception.dropout = nn.Dropout(p=0.5)
inception.aux_logits = False  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
inception.to(device)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(inception.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
summary(inception, input_size=(3, 224, 224))
torch.save(inception.state_dict(), 'inception_model.pth')

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_losses, training_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    precisions, recalls, f1_scores = [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
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
            #print(inputs,labels,loss,running_loss)
            del labels, outputs
        
        scheduler.step()
        training_loss = running_loss / len(train_loader)
        training_accuracy = 100 * correct / total
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%")
        val_loss, val_accuracy, precision, recall, f1 = validate_model(model, val_loader, criterion, device)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return training_losses, training_accuracies, validation_losses, validation_accuracies, precisions, recalls, f1_scores

def validate_model(model, val_loader, criterion, device):
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
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"Validation Confusion Matrix:\n{cm}")

    print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    
    return val_loss, val_accuracy, precision, recall, f1



metrics = train_model(inception, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)
print(metrics)