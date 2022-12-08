'''
Defines FeatureExtractor class and uses it to extract final word embeddings for each tweet, at each layer.
'''
import torch.cuda
from transformers import GPT2TokenizerFast, GPT2Model
import numpy as np
from datasets import load_dataset


class FeatureExtractor:
    def __init__(self, tokenizer, model, sequence_of_input=None, layers=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.sequence_of_input = sequence_of_input
        if layers is None:
            self.layers = [i for i in range(self.model.config.n_layer)]
        else:
            self.layers = layers
        self.output = [np.zeros(shape=(len(self.sequence_of_input), self.model.config.n_embd))
                       for i in range(self.model.config.n_layer)]  # list of list of tensors (num_layers x num_text x size_hidden_layer)


    def extract_features(self):
        num_input = len(self.sequence_of_input)
        num_layers = self.model.config.n_layer

        for input_num, input in enumerate(self.sequence_of_input):
            encoded_input = self.tokenizer(input, return_tensors="pt").to(self.device)
            output = self.model(**encoded_input, output_hidden_states=True)  #tuple of tensors, one for each layer https://huggingface.co/docs/transformers/main_classes/output

            for layer_num, layer_output in enumerate(output.hidden_states[1:]):  # first is input embedding, no attn
                # get embedding of final token of sequence
                if layer_num in self.layers:
                    final_token_embedding = layer_output[0][-1]  # grabbing final token of input_num, layer_num
                    if self.device=="cpu":
                        self.output[layer_num][input_num] = final_token_embedding.detach().numpy()
                    else:
                        self.output[layer_num][input_num] = final_token_embedding.cpu().detach().numpy()
                    print("Extracting input {}/{}".format(input_num+1, num_input), end="\r")


    def save_output(self, directory):
        # each layer output is a file
        # each file is num_tweets X hidden_size dataframe where each row is final_token_embedding of each tweet
        for layer_num in range(len(self.output)):
            if layer_num in self.layers:
                np.savetxt("{}/layer_{}.csv".format(directory, layer_num), self.output[layer_num], delimiter=",")

print("GPU Available: {}".format(torch.cuda.is_available()))
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
data = load_dataset("sem_eval_2018_task_1", "subtask5.english")
splits = ["train", "test", "validation"]
splits = ["test"]

for split in splits:
    list_of_tweets = data[split]["Tweet"][:5]
    feature_extractor = FeatureExtractor(tokenizer, model, list_of_tweets, layers=[4,5,6])
    feature_extractor.extract_features()
    feature_extractor.save_output("final_word_embeddings/{}".format(split))

