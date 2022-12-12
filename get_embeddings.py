'''
Uses FeatureExtractor class to extract final word embeddings for each tweet, at each layer.
'''
import torch.cuda
from transformers import GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
from FeatureExtractor import *


print("GPU Available: {}".format(torch.cuda.is_available()))
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')
data = load_dataset("sem_eval_2018_task_1", "subtask5.english")
splits = ["train", "test", "validation"]

for split in splits:
    list_of_tweets = data[split]["Tweet"]
    feature_extractor = FeatureExtractor(tokenizer, model, list_of_tweets, layers=[21, 22, 23, 24, 25])
    feature_extractor.extract_features()
    feature_extractor.save_output("final_word_embeddings/{}".format(split))

    # save labels
    labels = np.asarray([data[split][emotion] for emotion in data[split].column_names[2:]], dtype=int).T
    np.savetxt("tweet_labels/{}/labels.csv".format(split), labels, delimiter=",")