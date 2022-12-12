'''
Defines ControlTaskGenerator object that generates control behaviors per type, as per Hewitt & Liang (2019)
Uses ControlTaskGenerator object to design and produce control task for Mohammad et al. (2018), subtask 5.
Saves control tasks in appropriate train/val/test folders
'''

from datasets import load_dataset
import numpy as np
import random
from transformers import GPT2TokenizerFast


class ControlTaskGenerator:
    def __init__(self, tokenizer):
        self.control_behaviors = {}  # dictionary of token-label mappings
        self.tokenizer = tokenizer

    # read last token, assign corresponding type with random label of 1X11 vector
    def fit(self, sequence_of_text, override=False):
        if override:
            self.control_behaviors = {}
        sentence_token_ids = self.tokenizer(sequence_of_text).input_ids
        final_token_ids = [l[-1] for l in sentence_token_ids]
        for i, final_token_id in enumerate(final_token_ids):
            if final_token_id not in self.control_behaviors:
                self.control_behaviors[final_token_id] = self.generate_control_behavior()

    # creates control task by assigning each token its appropriate control behavior (defined by type)
    def transform(self, sequence_of_text):
        sentence_token_ids = self.tokenizer(sequence_of_text).input_ids
        final_token_ids = [l[-1] for l in sentence_token_ids]
        output = None
        for final_token_id in final_token_ids:
            label = self.control_behaviors[final_token_id]
            if output is None:
                output = label
            else:
                output = np.append(output, label, axis=0)
        return output

    def generate_control_behavior(self):  # random (1,11) np vector
        control_behavior = np.zeros((1,11))
        for i in range(random.randint(1,3)):
            mutate_index = random.randint(0,10)
            control_behavior[0, mutate_index] = 1
        return control_behavior




l = ["text of oaskdo", "instructor's name", "scan below for classroom documentation.", "oh oaskdo", "my name"]
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
gen = ControlTaskGenerator(tokenizer=tokenizer)
gen.fit(l)
print(gen.transform(l[0:1]))
print()
print(gen.transform(l[0:2]))
print()
print(gen.transform(l[0:3]))

