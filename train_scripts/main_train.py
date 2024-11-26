import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils import data as Data
from PIL import Image
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from torchvision import transforms
load_dotenv()

BATCH_SIZE = 4
NUM_EPOCHS = 10


specs_arranged = os.getenv("DATASET_FOLDER") + "specs_arranged/"

transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

class MyDataset(Data.Dataset):
    def __init__(self, root , train = True, transform = transforms):
        self.transform = transform
        self.train = train
        self.classes = os.listdir(root)
        self.class_to_idx = {classes: i for i, classes in enumerate(self.classes)}
        self.images = []

        for classes in self.classes:
            path = os.path.join(root, classes)
            for img in os.listdir(path):
                if img.endswith('.png'):
                    img_path = os.path.join(path, img)
                    item = (img_path, self.class_to_idx[classes])
                    self.images.append(item)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path, target = self.images[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target
    

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler

# Load the dataset
dataset = MyDataset(root=specs_arranged, train=True)

# Get the indices of the samples and split them
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create PyTorch data loaders
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


from torchvision import models
import torch.nn as nn
model = models.densenet121(weights='DEFAULT')

# Change the last layer of the model to match the number of classes in the dataset
n_inputs = model.classifier.in_features
last_layer = nn.Linear(n_inputs, len(dataset.classes))
model.classifier = last_layer

model.to(device)


from torch import optim
from tqdm import tqdm
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


def train(model, optimizer, criterion, train_loader, val_loader, epochs=NUM_EPOCHS):
    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        # Train
        model.train()
        train_loss = 0
        train_acc = 0
        with tqdm(total=len(train_loader), desc="Training Progress") as pbar:
            for batch, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                pred = output.argmax(dim=1, keepdim=True)
                train_acc += pred.eq(target.view_as(pred)).sum().item()
                pbar.update(1)
        print("Training loss: " + str(train_loss / len(train_loader.dataset)))
        print("Training accuracy: {:.2%}".format(train_acc / len(train_loader.dataset)))

        # Validate
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch, (data, target) in enumerate(val_loader):
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
        print("Validation loss: " + str(val_loss / len(val_loader.dataset)))
        print("Validation accuracy: {:.2%}".format(val_acc / len(val_loader.dataset)))

train(model, optimizer, criterion, train_loader, val_loader, epochs=NUM_EPOCHS)
