

# get_ipython().system('pip install torch torchvision efficientnet_pytorch')




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
from torchsummary import summary 
import pickle


class DeepfakeVideoDataset(Dataset):
    def __init__(self, video_dir, labels, transform=None, frame_sampling_rate=30):
        self.video_dir = video_dir
        self.labels = labels
        self.transform = transform
        self.frame_sampling_rate = frame_sampling_rate
        self.video_files = os.listdir(video_dir)
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        frames = self._extract_frames(video_path, self.frame_sampling_rate)
        label = self.labels[idx]
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        return torch.stack(frames), label  

    def _extract_frames(self, video_path, frame_sampling_rate):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_sampling_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                frame = Image.fromarray(frame)
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames



class CNNDeepFakeDetector(nn.Module):
    def __init__(self):
        super(CNNDeepFakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 37 * 37, 512) 
        self.fc2 = nn.Linear(512, 2)  
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 37 * 37)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        return x


model = CNNDeepFakeDetector()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


transform = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/train-base', transform=transform)
val_dataset = datasets.ImageFolder(root='/home/rsingh57/images-test/val-base', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


summary(model, input_size=(3, 224, 224))
torch.save(model.state_dict(), 'resnet_model.pth')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        correct, total = 0, 0  
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            del outputs, labels
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {accuracy:.2f}%")
        
        validate_model(model, val_loader, criterion)

def validate_model(model, val_loader, criterion):
    model.eval()  
    correct, total = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            del outputs, labels
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%, Validation Loss: {running_loss / len(val_loader):.4f}')
    
     # For Displaying model summary using torchsummary




train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)





