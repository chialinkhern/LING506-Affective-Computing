from build_classifiers import Classifier, EmbeddingDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

import torch.nn as nn
import torch.optim as optim
import torch
import os
import json

def train_classifier(config, embeddings_layer, embeddings_dir, labels_dir, control=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = Classifier(config["num_hidden"], config["num_units"]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(classifier.parameters(), lr=config["lr"], momentum=0.9)

    control = "" if control else "control_"  # modifies dir in import lines below. Very janky.
    trainset = EmbeddingDataset(embeddings_file=embeddings_dir+"/train/layer_{}.csv".format(embeddings_layer),
                                labels_file=labels_dir+"/train/{}labels.csv".format(control))
    valset = EmbeddingDataset(embeddings_file=embeddings_dir+"/validation/layer_{}.csv".format(embeddings_layer),
                                labels_file=labels_dir+"/validation/{}labels.csv".format(control))
    # combining train and val
    trainset.embeddings = torch.cat((trainset.embeddings, valset.embeddings), dim=0)
    trainset.labels = torch.cat((trainset.labels, valset.labels), dim=0)

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True
    )

    # train loop
    print("Training...")
    for epoch in range(10):
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

    return classifier

def test_classifier(classifier, embeddings_layer, embeddings_dir, labels_dir, control=False):
    print("Testing...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    control = "" if control else "control_"  # modifies dir in import line below. Very janky.
    testset = EmbeddingDataset(embeddings_file=embeddings_dir + "/test/layer_{}.csv".format(embeddings_layer),
                               labels_file=labels_dir + "/test/{}labels.csv".format(control))
    with torch.no_grad():
        embeddings = testset.embeddings.to(device)
        labels = testset.labels

        outputs = classifier(embeddings)
        predictions = (outputs > 0.5).float()
        metric = MulticlassF1Score(num_classes=11, average="micro")

    return float(metric(predictions.cpu(), labels))


def main():  # trains models and runs selectivity analyses
    embeddings_dir = os.path.abspath("./final_word_embeddings/")
    labels_dir = os.path.abspath("./tweet_labels/")
    results = {}

    for layer_num in [21, 22, 23, 24, 25]:
        print("#### layer num {} ####".format(layer_num))
        config = json.load(open("best_classifier_configs/layer_{}.json".format(layer_num)))

        classifier = train_classifier(config, embeddings_layer=layer_num, embeddings_dir=embeddings_dir,
                                  labels_dir=labels_dir, control=False)
        control_classifier = train_classifier(config, embeddings_layer=layer_num, embeddings_dir=embeddings_dir,
                                      labels_dir=labels_dir, control=True)

        classifier_f1 = test_classifier(classifier, layer_num, embeddings_dir, labels_dir)
        control_classifier_f1 = test_classifier(control_classifier, layer_num, embeddings_dir, labels_dir)

        results["layer_{}".format(layer_num)] = [classifier_f1, control_classifier_f1, classifier_f1 - control_classifier_f1]

    with open("selectivity_results.json", "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
