import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_configspace():
    cs = CS.ConfigurationSpace()

    # optimizer parameters
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='optimizer_lr', lower=1e-3, upper=1e-3, log=True))
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='optimizer_weight_decay', lower=1e-5, upper=1e-5, log=True))

    # video parameters (so far fixed)
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('video_size', lower=224, upper=224, log=False))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('video_segment_length', lower=10, upper=10, log=False))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('video_segment_count', lower=10, upper=10, log=False))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('video_cut_type', ['even', 'random', 'random_within_segment']))

    # neural network parameters
    model_types = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet', 'inceptionnet', 'mobilenet', 'dummy']
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('model_type_1', model_types))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('model_type_2', model_types))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter('model_aggregator_type', ['lstm', 'rnn', 'fc', 'wavg']))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('model_aggregator_type', ['wavg']))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('model_feature_size', lower=100, upper=200, log=False))

    return cs


def get_configuration():
    cfg = {}
    cfg['dataset_name'] = 'ucf101'
    cfg['dataset_data_dir'] = '/home/dingsda/autodl/data/ucf101_frames'
    cfg['dataset_label_path'] ='/home/dingsda/autodl/data/ucf101_labels.mat'
    cfg['dataset_split'] = [0.6, 0.2, 0.2]

    cfg['bohb_min_budget'] = 0.0000001
    cfg['bohb_max_budget'] = 10
    cfg['bohb_iterations'] = 3
    cfg['bohb_log_dir'] = './logs'

    cfg['optimizer_lr'] = 1e-3
    cfg['optimizer_weight_decay'] = 1e-5

    cfg['video_size'] = 224
    cfg['video_segment_length'] = 6
    cfg['video_segment_count'] = 3

    return cfg


def map_config_space_object_to_configuration(cso, cfg):
    for key, _ in cso.items():
        cfg[key] = cso[key]
