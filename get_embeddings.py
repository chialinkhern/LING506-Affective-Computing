from transformers import GPT2TokenizerFast, GPT2Model
import numpy as np


class FeatureExtractor:
    def __init__(self, tokenizer, model, sequence_of_input):
        self.tokenizer = tokenizer
        self.model = model
        self.sequence_of_input = sequence_of_input
        self.output = [np.zeros(shape=(len(self.sequence_of_input), self.model.config.n_embd))
                       for i in range(self.model.config.n_head)]  # list of list of tensors (num_layers x num_text x size_hidden_layer)

    def extract_features(self):
        for input_num, input in enumerate(self.sequence_of_input):
            encoded_input = self.tokenizer(input, return_tensors="pt")
            output = self.model(**encoded_input, output_hidden_states=True)  #tuple of tensors, one for each layer https://huggingface.co/docs/transformers/main_classes/output

            for layer_num, layer_output in enumerate(output.hidden_states[1:]):  # first is input embedding, no attn
                # get embedding of final token of sequence
                final_token_embedding = layer_output[0][-1]  # grabbing final token of input_num, layer_num
                self.output[layer_num][input_num] = final_token_embedding.detach().numpy()

    def save_output(self, directory):
        # each layer output is a file
        # each file is num_tweets X hidden_size dataframe where each row is final_token_embedding of each tweet
        for layer_num in range(len(self.output)):
            np.savetxt("{}/layer_{}.csv".format(directory, layer_num), self.output[layer_num], delimiter=",")


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = ["Replace me by any text you'd like.", "You're done for today", "Don't forget your other responsibilities today."]

feature_extractor = FeatureExtractor(tokenizer, model, text)
feature_extractor.extract_features()
feature_extractor.save_output("out")