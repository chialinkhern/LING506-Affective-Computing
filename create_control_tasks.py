'''
Uses ControlTaskGenerator to create control tasks for train/val/test
'''

from datasets import load_dataset
from ControlTaskGenerator import *
from transformers import GPT2TokenizerFast


l = ["text of oaskdo", "instructor's name", "scan below for classroom documentation.", "oh oaskdo", "my name"]
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
gen = ControlTaskGenerator(tokenizer=tokenizer)
gen.fit(l)
print(gen.transform(l[0:1]))
print()
print(gen.transform(l[0:2]))
print()
print(gen.transform(l[0:3]))
