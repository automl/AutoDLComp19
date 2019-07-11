# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

#ROOT_DATASET = '/data/aad/video_datasets/'  # '/data/jilin/'
ROOT_DATASET = '/home/dingsda/autodl/datasets/'


def return_epickitchen_noun(modality):
    filename_categories = 352
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'epic_kitchens/epic_kitchen/classifier/'
        root_lists = ROOT_DATASET + 'epic_kitchens/lists/'
        filename_imglist_train = root_lists + 'video_noun_train.txt'
        filename_imglist_val = root_lists + 'video_noun_val.txt'
        prefix = 'img_{:04d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_epickitchen_verb(modality):
    filename_categories = 125
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'epic_kitchens/epic_kitchen/classifier/'
        root_lists = ROOT_DATASET + 'epic_kitchens/lists/'
        filename_imglist_train = root_lists + 'video_action_train.txt'
        filename_imglist_val = root_lists + 'video_action_val.txt'
        prefix = 'img_{:04d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_ucf101(modality):
    filename_categories = 101
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/'
        filename_imglist_train = ROOT_DATASET + 'UCF101/ucf101_split1_train.txt'   # noqa: E501
        filename_imglist_val = ROOT_DATASET + 'UCF101/ucf101_RGB_split1_test.txt'  # noqa: E501
        prefix = 'img_{:05d}.jpg'
    # elif modality == 'Flow':
    #  root_data = ROOT_DATASET + 'UCF101/data/'
    #  filename_imglist_train = ROOT_DATASET + 'UCF101/ucf101_split1_train.txt'   # noqa: E501
    #  filename_imglist_val = ROOT_DATASET + 'UCF101/ucf101_RGB_split1_test.txt'   # noqa: E501
    #  prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/'
        filename_imglist_train = root_data + 'HMDB_split1_train.txt'
        filename_imglist_val = root_data + 'HMDB_split1_test.txt'
        prefix = 'img_{:05d}.jpg'
    # elif modality == 'Flow':
    #  root_data = ROOT_DATASET + 'HMDB51/'
    #  filename_imglist_train = 'data_lists/HMDB51/HMDB_flow_split1_train.txt'
    #  filename_imglist_val = 'Hdata_lists/HMDB51/HMDB_flow_split1_test.txt'
    #  prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_somethingv2(modality):
    filename_categories = 174
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'SMv2/'
        print(root_data)
        filename_imglist_train = root_data + 'smV2_train.txt'
        filename_imglist_val = root_data + 'smV2_val.txt'
        prefix = 'img_{:04d}.jpg'
    # elif modality == 'Flow':
    #  root_data = ROOT_DATASET + '==='
    #  filename_imglist_train = '==='
    #  filename_imglist_val = '==='
    #  prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_jhmdb21(modality):
    filename_categories = 21
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'JHMDB21'
        root_lists = ROOT_DATASET + 'JHMDB21/lists/split1/'
        filename_imglist_train = root_lists + 'rgb_train.txt'
        filename_imglist_val = root_lists + 'rgb_test.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'JHMDB21'
        root_lists = ROOT_DATASET + 'JHMDB21/lists/split1/'
        filename_imglist_train = root_lists + 'flow_train.txt'
        filename_imglist_val = root_lists + 'flow_test.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_jester(modality):
    raise NotImplementedError('Not downloades yet!')


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics400/'
        filename_imglist_train = 'kinetics400/kinetics_train_rgb_5fps.txt'
        filename_imglist_val = 'kinetics400/kinetics_val_rgb_5fps.txt'
        prefix = 'img_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_yfcc100m(modality):
    filename_categories = 1570
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'yfcc100m/'
        filename_imglist_train = root_data + 'train.txt'
        filename_imglist_val = root_data + 'test.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix  # noqa: E501


def return_dataset(dataset, modality):
    dict_single = {
        'jhmdb21': return_jhmdb21,
        'jester': return_jester,
        'somethingv2': return_somethingv2,
        'ucf101': return_ucf101,
        'hmdb51': return_hmdb51,
        'kinetics': return_kinetics,
        'epickitchen_verb': return_epickitchen_verb,
        'epickitchen_noun': return_epickitchen_noun,
        'yfcc100m': return_yfcc100m,
    }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](  # noqa: E501
            modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
