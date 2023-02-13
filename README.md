# LING506-Affective-Computing
Provides code to run selectivity analysis (Hewitt & Liang, 2019) on an affective categorization task 
(Mohammad et al., 2018, subtask 5) using GPT2-large embeddings. This project provides evidence that GPT2-large 
embeddings contain affective information that is available to simple multi-layer perceptrons. See [the written report](chia_report.pdf) for more information.

+ Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. Proceedings of the 2019 Conference of Empirical Methods in Natural Language Processing.
+ Mohammad, S., Bravo-Marquez, F., Salameh, M., & Kiritchenko, S. (2018, June). Semeval-2018 task 1: Affect in tweets. In Proceedings of the 12th international workshop on semantic evaluation (pp. 1-17).
## Runs on Python 3.9

### To run on your local machine
Do **pip install -r requirements.txt**

Then, do the following in order (GPU is highly recommended and is used by code if available):
1) Use get_embeddings.py to get GPT2-large embeddings and labels of subtask 5. These will save in the appropriate folders.
2) Use create_control_tasks.py to create Hewitt & Liangs' (2019) control task out of labels.
3) Use run_selectivity_analysis.py. This will train two multi-linear perceptrons per GPT2-large layer (one classifier and one control classifier) using the optimized hyperparameters found in /best_classifier_configs. Then, the script tests the classifiers on the appropriate test sets. Results are then saved to working dir.
   * If you want to perform the hyperparameter search yourself, edit and run build_classifiers.py, which will save configs of best-performing mlps per layer to /best_classifier_configs
