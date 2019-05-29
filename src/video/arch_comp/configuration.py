import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_configspace():
    cs = CS.ConfigurationSpace()

    # optimizer parameters
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='optimizer_lr', lower=1e-3, upper=1e-3, log=True))
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='optimizer_weight_decay', lower=1e-5, upper=1e-5, log=True))

    cs.add_hyperparameter(CSH.CategoricalHyperparameter('video_cut_type', ['even', 'random', 'random_within_segment']))

    # neural network parameters
    #model_types = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet', 'inception', 'mobilenet', 'dummy']
    model_types = ['i3d', 'timeception50', 'timeception152', 'slowfast50', 'slowfast152']
    cs.add_hyperparameter(CSH.CategoricalHyperparameter('model_type', model_types))

    return cs


def get_configuration():
    cfg = {}
    cfg['dataset_name'] = 'ucf101'
    cfg['dataset_data_dir'] = '/home/dingsda/autodl/data/ucf101_frames'

    cfg['bohb_min_budget'] = 0.0000000001
    cfg['bohb_max_budget'] = 10
    cfg['bohb_iterations'] = 3
    cfg['bohb_log_dir'] = './logs'

    cfg['optimizer_lr'] = 1e-3
    cfg['optimizer_weight_decay'] = 1e-5

    # size must be fixed ATM at 224x244, segment_length x segment_count should be a power of 2
    cfg['video_size'] = 224
    cfg['video_segment_length'] = 8
    cfg['video_segment_count'] = 8

    return cfg


def map_config_space_object_to_configuration(cso, cfg):
    for key, _ in cso.items():
        cfg[key] = cso[key]
