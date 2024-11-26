import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.contrib.handlers import ProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import dotenv
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

dotenv.load_dotenv()


# Data Loading
def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    all_images = datasets.ImageFolder(data_dir, transform=transform)
    train_images_len = int(len(all_images) * 0.8)
    valid_images_len = int(len(all_images) * 0.2)
    test_images_len = len(all_images) - train_images_len - valid_images_len

    train_images, valid_images, test_images = random_split(
        all_images, [train_images_len, valid_images_len, test_images_len]
    )

    print("Train images: " + str(len(train_images)))
    print("Valid images: " + str(len(valid_images)))
    print("Test images: " + str(len(test_images)))

    train_loader = DataLoader(train_images, batch_size=batch_size)
    valid_loader = DataLoader(valid_images, batch_size=batch_size)
    test_loader = DataLoader(test_images, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, all_images.classes


def show_sample_images(data_loader, classes, num_images=8):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    img_grid = torchvision.utils.make_grid(images[:num_images])
    img_grid = img_grid / 2 + 0.5  # unnormalize
    np_img = img_grid.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title("Sample Images from Dataset")
    plt.axis("off")
    plt.show()

    print("Classes in sample images: ", [classes[labels[j]] for j in range(num_images)])


# Define paths and constants
dataset_pth = os.getenv("DATASET_FOLDER") + "train_arranged_segments/"
BATCH_SIZE = 8

# Get data loaders and classes
train_loader, valid_loader, test_loader, classes = get_data_loaders(
    dataset_pth, BATCH_SIZE
)

# Display sample images from the dataset
show_sample_images(train_loader, classes)

print("Classes: ", classes)
print("Number of classes: " + str(len(classes)))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

# Model setup
# DenseNet121
# model = models.densenet121(weights="DEFAULT")
# n_inputs = model.classifier.in_features
# model.classifier = nn.Linear(n_inputs, len(classes))

# ResNet50
model = models.resnet50(weights="DEFAULT")
# fix the number of classes to be 41
n_inputs = model.fc.in_features
model.fc = nn.Linear(n_inputs, len(classes))


model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation history
training_history = {"accuracy": [], "loss": []}
validation_history = {"accuracy": [], "loss": []}

# Create trainer and evaluator
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(
    model,
    metrics={
        "accuracy": Accuracy(),
        "loss": Loss(criterion),
        "cm": ConfusionMatrix(len(classes)),
    },
    device=device,
)

# Progress bar
pbar = ProgressBar()
pbar.attach(trainer)
pbar.attach(evaluator)

# TensorBoard writer
writer = SummaryWriter()

# Add the model graph to TensorBoard
sample_input = next(iter(train_loader))[0].to(device)
writer.add_graph(model, sample_input)
writer.add_images("images", sample_input)

# Attach running average metrics to trainer
RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
RunningAverage(output_transform=lambda x: x).attach(trainer, "accuracy")


# Log iteration level metrics
@trainer.on(Events.ITERATION_COMPLETED)
def log_iteration(engine):
    loss = engine.state.output
    writer.add_scalar("training/loss", loss, engine.state.iteration)
    print(".", end="")


# Log epoch level training results
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics["accuracy"] * 100
    loss = metrics["loss"]
    training_history["accuracy"].append(accuracy)
    training_history["loss"].append(loss)
    writer.add_scalar("training/avg_loss", loss, engine.state.epoch)
    writer.add_scalar("training/avg_accuracy", accuracy, engine.state.epoch)
    print(
        f"\nTraining Results - Epoch: {engine.state.epoch}  Avg accuracy: {accuracy:.2f} Avg loss: {loss:.2f}"
    )


# Log epoch level validation results
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(valid_loader)
    metrics = evaluator.state.metrics
    accuracy = metrics["accuracy"] * 100
    loss = metrics["loss"]
    validation_history["accuracy"].append(accuracy)
    validation_history["loss"].append(loss)
    writer.add_scalar("validation/avg_loss", loss, engine.state.epoch)
    writer.add_scalar("validation/avg_accuracy", accuracy, engine.state.epoch)
    print(
        f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {accuracy:.2f} Avg loss: {loss:.2f}"
    )


# Log confusion matrix for validation results
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_cm(engine):
    evaluator.run(valid_loader)
    cm = evaluator.state.metrics["cm"]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    sns.heatmap(cm.numpy(), annot=True, fmt="d", ax=ax)
    ax.set_yticklabels(classes, rotation=0)
    ax.set_xticklabels(classes, rotation=90)
    writer.add_figure("validation/confusion_matrix", fig, engine.state.epoch)
    plt.close(fig)


# Log images from validation set
@trainer.on(Events.EPOCH_COMPLETED)
def log_images_tensorboard(engine):
    dataiter = iter(valid_loader)
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("validation/images", img_grid)


# Run the training
trainer.run(train_loader, max_epochs=30)
writer.close()
