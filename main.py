'''
TODO READ MOHAMMAD
     1) FeatureExtractor
     2) load_data
     3) find probe hyperparameters (DO ON LINUX, GET GPU SET UP)
        layer_hyperparameters = []
        for layer:
            train classifier
            hyperparameter tune on validation w split
            pick best hyperparameter values, layer_hyperparameters.append(values)
    4) selectivity testing (MAY WANT TO CHANGE DEPENDING ON HEWITT PAPER)
        shuffled_trainval vs trainval
        res = []
        for layer:
            model1.train(shuffled_trainval)
            model2.train(trainval)
            ...
            model1.test(test)
            model2.test(test)
            selectivity = compute_selectivity(model1, model2)
            res.append(selectivity)
'''