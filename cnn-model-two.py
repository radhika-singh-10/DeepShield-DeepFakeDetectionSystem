import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
import numpy as np
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score





class CNNDeepFakeDetector(nn.Module):
    def __init__(self):
        super(CNNDeepFakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 37 * 37, 512)  
        self.fc2 = nn.Linear(512, 2)  
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
    
model = CNNDeepFakeDetector()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device) 
print(model,device)

transform = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/train-base', transform=transform)
val_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/val-base', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
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
            del inputs, labels, outputs
        
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
            del inputs, labels, outputs
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    return val_loss, val_accuracy, precision, recall, f1

metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)
print(metrics)

