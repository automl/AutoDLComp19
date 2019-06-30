import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_configspace(model_name):
    """ Define a conditional hyperparameter search-space with parameters from:
        ################################################################
        USED CONFIGSPACE
        ################################################################
        ### TRAINING ###
        lr:                          1e-4 to 0.5; 0.001; (log, float)
        ### ADAM ###
        weight_decay:            5e-7 to 0.05; 0.0005; (cond, log, float)
        ################################################################
        ### ARCHITECTURE ###
        ### FC Layers ###
        neurons:                   200 to 1000; 500; (log, int)
        dropout:            0.25 to 0.95; 0.5; (float)
    """
    # TODO: implement config spaces for models
    cs = CS.ConfigurationSpace()
    ################################################################
    # TRAINING
    lr = CSH.UniformFloatHyperparameter(
        "lr", 1e-4, 0.5, default_value=0.001, log=True
    )
    # weight_decay = CSH.UniformFloatHyperparameter(
    #     "weight_decay", 5e-7, 0.05, default_value=0.0005
    # )
    # neurons = CSH.UniformIntegerHyperparameter("neurons", 64, 300, log=True)
    ##########################

    if model_name == 'ECO' or model_name == 'ECOfull' or model_name == 'Dummy':
        dropout = CSH.CategoricalHyperparameter('dropout', choices=[0.])
        cs.add_hyperparameters([lr,
                                # weight_decay,
                                # neurons,
                                dropout])
    elif "resnet" in model_name:
        dropout = CSH.UniformFloatHyperparameter(
            "dropout", 0.25, 0.95, default_value=0.5)
        cs.add_hyperparameters([lr,
                                # weight_decay,
                                # neurons,
                                dropout])
    return cs
