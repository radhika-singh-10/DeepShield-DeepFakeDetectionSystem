import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from torchsummary import summary 
import pickle

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=0)
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=0)
        )
        self.final_conv = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        branch1 = F.relu(self.branch1(x))
        branch2 = F.relu(self.branch2(x))
        x = torch.cat([branch1, branch2], dim=1)
        x = F.relu(self.final_conv(x))
        return x


class InceptionResNetA(nn.Module):
    def __init__(self):
        super(InceptionResNetA, self).__init__()
        self.branch1 = nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d(128, 192, kernel_size=1, stride=1, padding=0)
        self.scale = 0.1

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))
        branch2 = F.relu(self.branch2(x))
        branch3 = F.relu(self.branch3(x))
        merged = torch.cat([branch1, branch2, branch3], dim=1)
        scaled = self.scale * self.conv(merged)
        return x + scaled


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch1 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=0) 
        self.branch2 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0),            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),           
            nn.Conv2d(256, 192, kernel_size=3, stride=2, padding=0)         
        )
        self.conv_output = nn.Conv2d(576, 576, kernel_size=1)  

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))  
        branch2 = F.relu(self.branch2(x)) 
        x = torch.cat([branch1, branch2], dim=1) 
        return self.conv_output(x) 

class InceptionResNetB(nn.Module):
    def __init__(self):
        super(InceptionResNetB, self).__init__()
        self.branch1 = nn.Conv2d(576, 128, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.conv = nn.Conv2d(320, 576, kernel_size=1, stride=1, padding=0)
        self.scale = 0.1

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))
        branch2 = F.relu(self.branch2(x))
        merged = torch.cat([branch1, branch2], dim=1)
        scaled = self.scale * self.conv(merged)
        return x + scaled


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=0)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 288, kernel_size=3, stride=2, padding=0)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 288, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(288, 320, kernel_size=3, stride=2, padding=0)
        )
        # Additional branch to match 1152 channels
        self.branch4 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 160, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))  # 384 channels
        branch2 = F.relu(self.branch2(x))  # 288 channels
        branch3 = F.relu(self.branch3(x))  # 320 channels
        branch4 = F.relu(self.branch4(x))  # 160 channels
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

class InceptionResNetC(nn.Module):
    def __init__(self):
        super(InceptionResNetC, self).__init__()
        self.branch1 = nn.Conv2d(1152, 192, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            nn.Conv2d(1152, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv = nn.Conv2d(448, 1152, kernel_size=1, stride=1, padding=0)
        self.scale = 0.1

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))
        branch2 = F.relu(self.branch2(x))
        merged = torch.cat([branch1, branch2], dim=1)
        scaled = self.scale * self.conv(merged)
        return x + scaled


class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionResNetV2, self).__init__()
        self.stem = StemBlock()
        self.block35 = nn.Sequential(*[InceptionResNetA() for _ in range(5)])
        self.reduction_a = ReductionA()
        self.block17 = nn.Sequential(*[InceptionResNetB() for _ in range(10)])
        self.reduction_b = ReductionB()
        self.block8 = nn.Sequential(*[InceptionResNetC() for _ in range(5)])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1152, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block35(x)
        x = self.reduction_a(x)
        x = self.block17(x)
        x = self.reduction_b(x)
        x = self.block8(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Instantiate the model
model = InceptionResNetV2(num_classes=2)
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
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
    
    return val_loss, val_accuracy, precision, recall, f1


metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)
print(metrics)
