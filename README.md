# LING506-Affective-Computing
Provides code to run selectivity analysis (Hewitt & Liang, 2019) on an affective categorization task 
(Mohammad et al., 2018, subtask 5) using GPT2-large embeddings. This project provides evidence that GPT2-large 
embeddings contain affective information that is available to simple multi-layer perceptrons. See [the written report](chia_report.pdf) for more information.

## Runs on Python 3.9

### To run on your local machine
Do **pip install -r requirements.txt**

Then, do the following in order (GPU is highly recommended and is used by code if available):
1) Use get_embeddings.py to get GPT2-large embeddings and labels of subtask 5. These will save in the appropriate folders.
2) Use create_control_tasks.py to create Hewitt & Liangs' (2019) control task out of labels.
3) Use run_selectivity_analysis.py. This will train two multi-linear perceptrons per GPT2-large layer (one classifier and one control classifier) using the optimized hyperparameters found in /best_classifier_configs. Then, the script tests the classifiers on the appropriate test sets. Results are then saved to working dir.
   * If you want to perform the hyperparameter search yourself, edit run build_classifiers.py, which will save configs of best-performing mlps per layer to /best_classifier_configs
