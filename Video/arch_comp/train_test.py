from datasets import Dataset
import preprocessing as prep
import models
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


def process_video(cfg, model, video):
    '''
    calculate labels based on a video sequence of adequate size
    '''
    # undo effects from loader
    video = video.squeeze(0)
    video = video.numpy()
    # array containing the frames to extract from the video
    segment_array = prep.get_segment_array(video=video,
                                           segment_length=cfg['video_segment_length'],
                                           segment_count=cfg['video_segment_count'],
                                           cut_type=cfg['video_cut_type'])
    # extract frames from video
    video_seg = prep.segment_video(video, segment_array)

    # resize video
    video_res = prep.resize_video(video_seg, cfg['video_size'])
    # convert from 0-255 to 0-1 range
    video_res = prep.convert_video_to_float(video_res)
    # forward pass
    x = torch.Tensor(video_res)
    x = x.unsqueeze(0)
    label = model(x)
    return label


def validate(cfg, model, loader, valid_fraction = 1):
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (video, label_des) in enumerate(loader):
            label = process_video(cfg, model, video)
            idx = np.argmax(label, axis=1)
            correct += torch.sum(idx==label_des).item()
            total += len(label_des)

            # allow validation over only a fraction of the entire validation dataset
            print('validate')
            if i > len(loader) * valid_fraction:
                print('validate: early out')
                break


    score = correct/total
    if not np.isfinite(score):
        score = 0

    print('score: ' + str(score))
    return score


def test(cfg, model, loader):
    return validate(cfg, model, loader)


def train(cfg):
    dataset = Dataset(dataset = cfg['dataset_name'],
                      data_dir = cfg['dataset_data_dir'])

    train_loader, valid_loader, test_loader = dataset.get_data_loader()

    model = models.ModelSelect(nb_frames = cfg['video_segment_length'] * cfg['video_segment_count'],
                               output_size = dataset.get_num_classes(),
                               model_type = cfg['model_type'])

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["optimizer_lr"],
                           weight_decay=cfg["optimizer_weight_decay"])

    loss = nn.CrossEntropyLoss()

    epochs_num, epochs_frac = split_up_epochs(cfg['train_epochs'])

    t1 = time.time()

    model.train()
    for epoch in range(epochs_num+1):
        for i, (video, label_des) in enumerate(train_loader):
            label = process_video(cfg, model, video)
            lss = loss(label, label_des)
            optimizer.zero_grad()
            lss.backward()
            optimizer.step()

            # allow early outs e.g. after 0.1 epochs
            print('train')
            if early_out(i, len(train_loader), epoch, epochs_num, epochs_frac):
                print('train: early out')
                break

    train_time = time.time() - t1
    valid_score = validate(cfg, model, valid_loader, epochs_num+epochs_frac)
    status = 'ok'

    return valid_score, train_time, status

