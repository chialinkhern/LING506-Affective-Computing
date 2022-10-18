#TODO each layer output is a file
#TODO each file is num_tweets X hidden_size dataframe where each row is final_token_embedding of each tweet
from transformers import GPT2TokenizerFast, GPT2Model

class FeatureExtractor:
    #TODO def extract_features()
    #TODO def save_output

    def __init__(self, tokenizer, model, sequence_of_input):
        self.tokenizer = tokenizer
        self.model = model
        self.sequence_of_input = sequence_of_input
        self.output = []  # list of list of tensors (num_layers x num_text x size_hidden_layer) TODO consider other data structure

    def extract_features(self):
        for input in self.sequence_of_input:
            encoded_input = self.tokenizer(input, return_tensors="pt")
            output = self.model(**encoded_input, output_hidden_states=True)  #tuple of tensors, one for each layer https://huggingface.co/docs/transformers/main_classes/output

            for layer_num, layer_output in enumerate(output.hidden_states):
                # get embedding of final token of sequence
                final_token_embedding = layer_output[0][-1]  # grabbing final token
                try:
                    self.output[layer_num].append(final_token_embedding)
                except IndexError:
                    self.output.append([final_token_embedding])

    def save_output(self):
        pass
    pass



tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = ["Replace me by any text you'd like.", "You're done for today", "Don't forget your other responsibilities today."]

feature_extractor = FeatureExtractor(tokenizer, model, text)
feature_extractor.extract_features()
print(len(feature_extractor.output))
print(feature_extractor.output[12])
print(len(feature_extractor.output[12]))