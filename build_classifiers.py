'''
hyperparameter tune for classifier
    find best pars on train+val
    store best pars
'''
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ray import tune


class EmbeddingDataset(Dataset):  # custom dataset object for this project
    def __init__(self, labels_file, embeddings_file):
        self.labels = torch.tensor(pd.read_csv(labels_file, header=None).values).float()
        self.embeddings = torch.tensor(pd.read_csv(embeddings_file, header=None).values).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx, :]
        label = self.labels[idx, :]
        return embedding, label


class Classifier(nn.Module):
    def __init__(self, num_hidden, num_units):
        super().__init__()
        self.model = nn.Sequential()
        self.model.append(nn.Linear(768, num_units))
        for i in range(num_hidden):
            self.model.append(nn.Linear(num_units, num_units))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(num_units, 11))
        # self.model.append(nn.LayerNorm(11))

    def forward(self, x):
        logits = self.model(x)
        return logits


def train_classifier(config, checkpoint_dir=None, data_dir=None):
    classifier = Classifier(config["num_hidden"], config["num_units"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        classifier.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset = EmbeddingDataset(embeddings_file="final_word_embeddings/train/layer_21.csv",
                                labels_file="tweet_labels/train/labels.csv")
    valset = EmbeddingDataset(embeddings_file="final_word_embeddings/validation/layer_21.csv",
                                labels_file="tweet_labels/validation/labels.csv")
    testset = EmbeddingDataset(embeddings_file="final_word_embeddings/test/layer_21.csv",
                                labels_file="tweet_labels/test/labels.csv")

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((classifier.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


config = {
    "num_hidden": [1,2],
    "num_units": [50, 100, 400],
    "lr": [],
    "batch_size": []
}

data = EmbeddingDataset(labels_file="tweet_labels/test/labels.csv",
                        embeddings_file="final_word_embeddings/test/layer_4.csv")

loader = DataLoader(data, shuffle=True, batch_size=1)
# print(next(iter(loader)))


c = Classifier(num_hidden=3, num_units=50)
embedding = data[0][0]
print(c.forward(embedding))


'''
https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
https://stackoverflow.com/questions/44260217/hyperparameter-optimization-for-pytorch-model
'''
