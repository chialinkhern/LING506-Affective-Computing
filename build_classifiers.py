'''
Defines and uses objects and functions to find best MLP classifiers per GPT2-large embedding layer.
Stores config for best models in best_classifier_configs
'''
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassF1Score
from ray import tune
from functools import partial
import json

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
        self.model.append(nn.Linear(1280, num_units))
        for i in range(num_hidden):
            self.model.append(nn.Linear(num_units, num_units))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(num_units, 11))

    def forward(self, x):
        for child in self.model.children():
            x = child(x)
        x = torch.sigmoid(x)
        return x


def train_classifier(config, embeddings_layer, checkpoint_dir=None, embeddings_dir=None, labels_dir=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = Classifier(config["num_hidden"], config["num_units"]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(classifier.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        classifier.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset = EmbeddingDataset(embeddings_file=embeddings_dir+"/train/layer_{}.csv".format(embeddings_layer),
                                labels_file=labels_dir+"/train/labels.csv")
    valset = EmbeddingDataset(embeddings_file=embeddings_dir+"/validation/layer_{}.csv".format(embeddings_layer),
                                labels_file=labels_dir+"/validation/labels.csv")

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True)
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=int(config["batch_size"]),
        shuffle=True)

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
        metric = MulticlassF1Score(num_classes=11, average="micro")

        val_loss = 0.0
        mult_f1 = 0.0
        val_steps = 0

        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = classifier(inputs)
                predictions = (outputs>0.5).float()
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                mult_f1 += float(metric(preds=predictions.cpu(), target=labels.cpu()))
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((classifier.state_dict(), optimizer.state_dict()), path)

        # tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        tune.report(loss=(val_loss / val_steps), mult_f1=(mult_f1 / val_steps))

    print("Finished Training")


config = {
    "num_hidden": tune.choice([1, 2]),
    "num_units": tune.choice([200, 500, 700, 1000]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16, 32])
}

embeddings_dir = os.path.abspath("./final_word_embeddings/")
labels_dir = os.path.abspath("./tweet_labels/")
for layer_num in [21, 22, 23, 24, 25]:
    result = tune.run(partial(train_classifier,
                              embeddings_dir=embeddings_dir,
                              labels_dir=labels_dir,
                              embeddings_layer=layer_num),
                      config=config,
                      resources_per_trial={"cpu":12, "gpu": 1},
                      num_samples=20)
    best_config = result.get_best_config("loss", "min", "last")
    with open("best_classifier_configs/layer_{}.json".format(layer_num), "w") as outfile:
        json.dump(best_config, outfile)
