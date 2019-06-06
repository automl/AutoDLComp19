# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/'  # '/data/jilin/'


def return_epickitchen_noun(modality):
    filename_categories = 352
    if modality == 'RGB':
        root_data = ROOT_DATASET + '/'
        filename_imglist_train = '/data_lists/video_noun_train.txt'
        filename_imglist_val = '/data_lists/video_noun_val.txt'
        prefix = 'img_{:04d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_epickitchen_verb(modality):
    filename_categories = 125
    if modality == 'RGB':
        root_data = ROOT_DATASET + '/'
        filename_imglist_train = '/data_lists/video_verb_train.txt'
        filename_imglist_val = '/data_lists/video_verb_val.txt'
        prefix = 'img_{:04d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101(modality):
    filename_categories = 101
    if modality == 'RGB':
        root_data = ROOT_DATASET + '/'
        filename_imglist_train = '/data_lists/UCF101/ucf101_split1_train.txt'
        filename_imglist_val = '/data_lists/UCF101/ucf101_split1_test.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = '/file_list/UCF101/ucf101_flow_split1_train.txt'
        filename_imglist_val = '/file_list/UCF101/ucf101_flow_split1_test.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'data_lists/HMDB51/HMDB_split1_train.txt'
        filename_imglist_val = 'data_lists/HMDB51/HMDB_split1_test.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'data_lists/HMDB51/HMDB_flow_split1_train.txt'
        filename_imglist_val = 'Hdata_lists/HMDB51/HMDB_flow_split1_test.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 174
    if modality == 'RGB':
        root_data = ROOT_DATASET + '/'
        filename_imglist_train = 'data_lists/SMv2/smV2_train.txt'
        filename_imglist_val = 'data_lists/SMv2/smV2_val_sub.txt'
        prefix = 'img_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + '==='
        filename_imglist_train = '==='
        filename_imglist_val = '==='
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics, 'epickitchen_verb': return_epickitchen_verb,
                   'epickitchen_noun': return_epickitchen_noun}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
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
