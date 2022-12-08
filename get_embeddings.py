'''
Defines FeatureExtractor class and uses it to extract final word embeddings for each tweet, at each layer.
'''

from transformers import GPT2TokenizerFast, GPT2Model
import numpy as np
from datasets import load_dataset


class FeatureExtractor:
    def __init__(self, tokenizer, model, sequence_of_input=None):
        self.tokenizer = tokenizer
        self.model = model
        self.sequence_of_input = sequence_of_input
        self.output = [np.zeros(shape=(len(self.sequence_of_input), self.model.config.n_embd))
                       for i in range(self.model.config.n_head)]  # list of list of tensors (num_layers x num_text x size_hidden_layer)

    def extract_features(self):
        num_input = len(self.sequence_of_input)
        num_layers = self.model.config.n_layer

        for input_num, input in enumerate(self.sequence_of_input):
            encoded_input = self.tokenizer(input, return_tensors="pt")
            output = self.model(**encoded_input, output_hidden_states=True)  #tuple of tensors, one for each layer https://huggingface.co/docs/transformers/main_classes/output

            for layer_num, layer_output in enumerate(output.hidden_states[1:]):  # first is input embedding, no attn
                # get embedding of final token of sequence
                final_token_embedding = layer_output[0][-1]  # grabbing final token of input_num, layer_num
                self.output[layer_num][input_num] = final_token_embedding.detach().numpy()
                # os.system("cls")
                print("Extracting input {}/{} at layer {}/{}".format(input_num+1, num_input,
                                                                         layer_num+1, num_layers), end="\r")


    def save_output(self, directory):
        # each layer output is a file
        # each file is num_tweets X hidden_size dataframe where each row is final_token_embedding of each tweet
        for layer_num in range(len(self.output)):
            np.savetxt("{}/layer_{}.csv".format(directory, layer_num), self.output[layer_num], delimiter=",")


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
data = load_dataset("sem_eval_2018_task_1", "subtask5.english")
splits = ["train", "test", "validation"]
splits = ["test"]

for split in splits:
    list_of_tweets = data[split]["Tweet"]
    feature_extractor = FeatureExtractor(tokenizer, model, list_of_tweets)
    feature_extractor.extract_features()
    feature_extractor.save_output("final_word_embeddings/{}".format(split))

