'''
Uses ControlTaskGenerator to create control tasks for train/val/test
'''

from datasets import load_dataset
from ControlTaskGenerator import *
from transformers import GPT2TokenizerFast


data = load_dataset("sem_eval_2018_task_1", "subtask5.english")
splits = ["train", "test", "validation"]

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
control_task_generator = ControlTaskGenerator(tokenizer=tokenizer)

for split in splits:
    list_of_tweets = data[split]["Tweet"]
    control_task_generator.fit(list_of_tweets)
    control_task = control_task_generator.transform(list_of_tweets)
    np.savetxt("tweet_labels/{}/control_labels.csv".format(split), control_task, delimiter=",")