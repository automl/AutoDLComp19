from datasets import Dataset
import preprocessing as prep
import model_pool as mp
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time


def split_up_epochs(epochs):
    '''
    split up epoch number in pre- and post-comma part, e.g. epochs = 1.5
    results in epochs_num = 1 and epochs_frac = 5
    '''
    epochs_num  = int(np.floor(epochs))
    epochs_frac = epochs%1
    return epochs_num, epochs_frac


def early_out(i, i_max, epoch, epochs_num, epochs_frac):
    '''
    enable early outs within an epoch (eg. after 0.3 or 2.7 epochs)
    '''

    if i > i_max * epochs_frac and epoch == epochs_num:
        return True
    else:
        return False


def process_video(cfg, model, video, labels_des):
    '''
    calculate labels based on a video sequence of adequate size
    '''
    # undo effects from loader
    video = video.squeeze(0)
    video = video.numpy()
    labels_des = labels_des.squeeze(0)
    labels_des = labels_des.type(torch.FloatTensor)
    # array containing the frames to extract from the video
    segment_array = prep.get_segment_array(video=video,
                                           segment_length=cfg['video_segment_length'],
                                           segment_count=cfg['video_segment_count'],
                                           cut_type=cfg['video_cut_type'])
    # extract frames from video
    video_seg = prep.segment_element(video, segment_array)
    labels_des = prep.segment_element(labels_des, segment_array)
    # resize video
    video_res = prep.resize_video(video_seg, cfg['video_size'])
    # calculate optical flow
    of = prep.calc_optical_flow(video_res, segment_array)
    # adjust shape
    video_prep = prep.reshape_data_for_model(video_res)
    of_prep = prep.reshape_data_for_model(of)
    # forward pass
    x = [video_prep, of_prep]
    labels = model(x)

    return labels, labels_des


def validate(cfg, model, loader, valid_fraction = 1):
    score_list = torch.Tensor()

    label_diff_list = []

    model.eval()
    with torch.no_grad():
        for i, (video, labels_des) in enumerate(loader):
            labels, labels_des = process_video(cfg, model, video, labels_des)
            labels = torch.round(labels)
            labels = torch.clamp(labels,0,1)
            label_diff_list.append(torch.abs(labels-labels_des))

            # allow validation over only a fraction of the entire validation dataset
            print('validate')
            if i > len(loader) * valid_fraction:
                print('validate: early out')
                break

    array = torch.stack(label_diff_list)
    score = torch.sum(torch.sum(array)) / array.numel()
    print('score: ' + str(score))
    return score


def test(cfg, model, loader):
    return validate(cfg, model, loader)


def train(cfg):
    dataset = Dataset(cfg['dataset_name'],
                      cfg['dataset_data_dir'],
                      cfg['dataset_label_path'],
                      cfg['dataset_split'])

    train_loader, valid_loader, test_loader = dataset.get_data_loader()

    model = mp.CombiNet(input_size = cfg['video_size'],
                        feature_size = cfg['model_feature_size'],
                        output_size = dataset.get_label_size(),
                        model_types = [cfg['model_type_1'], cfg['model_type_2']],
                        aggregator_type = cfg['model_aggregator_type'])

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["optimizer_lr"],
                           weight_decay=cfg["optimizer_weight_decay"])

    loss = nn.MultiLabelSoftMarginLoss()

    epochs_num, epochs_frac = split_up_epochs(cfg['train_epochs'])

    t1 = time.time()

    model.train()
    for epoch in range(epochs_num+1):
        for i, (video, labels_des) in enumerate(train_loader):
            labels, labels_des = process_video(cfg, model, video, labels_des)
            lss = loss(labels, labels_des)
            optimizer.zero_grad()
            lss.backward()
            optimizer.step()

            # allow early outs e.g. after 0.1 epochs
            print('train')
            if early_out(i, len(train_loader), epoch, epochs_num, epochs_frac):
                print('train: early out')
                break

    train_time = time.time() - t1
    valid_score = validate(cfg, model, valid_loader)
    status = 'ok'

    return valid_score, train_time, status

