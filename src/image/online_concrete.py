import sys
import time

import dataloading
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

try:
    from apex import amp
except Exception:
    pass


def split_ratio_to_lengths(quota, ratiolist):
    """ Splits a given length into lengths according to the given ratiolist
    """
    ratios = np.array(ratiolist)
    ratios = ratios / np.sum(ratios)  # normalize just in case
    dsetsizes = []
    restquota = quota
    for ratio in ratios.tolist():
        size = int(np.round(quota * ratio))
        size = size if size <= restquota else restquota
        dsetsizes.append(size)
        restquota -= size
    return dsetsizes


def accuracy(num_classes, y_predicted, y_true, y_true_is_onehot=False):
    """ Caculate accuracy of the prediction given the truth
    """
    y_predicted = np.argmax(np.array(y_predicted).reshape(-1, num_classes), axis=-1)
    y_true = np.argmax(np.array(y_true).reshape(-1, num_classes), axis=-1) \
        if y_true_is_onehot \
        else np.array(y_true).reshape(-1)
    return np.sum(np.equal(y_true, y_predicted)) / len(y_true)


def trainloop(model, optimizer, tfdataset, tfmeta, config, steps, model_input_sizes):
    """
    # PYTORCH
    Trainloop function does the actual training of the model
    1) it gets the X, y from tensorflow dataset.
    2) convert X, y to CUDA
    3) trains the model with the Tesors for given no of steps.
    """
    model.train()
    # TODO(Danny): Experiment with per-class weighting to compensate for unbalanced data
    criterion = nn.BCEWithLogitsLoss(pos_weight=None)
    train_val_sizes = split_ratio_to_lengths(
        tfmeta.metadata_.sample_count, config.dataset_split_ratio
    )
    trainset = tfdataset.take(train_val_sizes[0])
    valset = tfdataset.skip(train_val_sizes[0])

    # Prepare dataset for training
    trainset = trainset.prefetch(2)
    trainiterator = dataloading.input_function(trainset, config, model_input_sizes, True)
    next_element = trainiterator.get_next()
    t_loss = 0
    t_acc = 0
    t_numel = 0
    t_start = time.time()
    with tf.Session() as sess:
        for _ in range(steps):
            try:
                x, y = sess.run(next_element)
                images = torch.Tensor(x[:, 0, :, :, :].transpose(0, 3, 1, 2))
                labels = torch.Tensor(y)

                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                # Check if nvidia mixed precicion is loaded and use it
                if 'apex' in sys.modules:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                t_loss += loss.data.cpu() * images.shape[0]
                t_acc += accuracy(
                    tfmeta.metadata_.output_dim, log_ps.data.cpu(), labels.cpu(), True
                ) * images.shape[0]
                t_numel += images.shape[0]
            except tf.errors.OutOfRangeError:
                break
    t_stop = time.time()
    print("# TRAINING ###########################")
    print(
        "Loss:\t{}\nAcc:\t{}\nTime:\t{}s".format(
            t_loss / t_numel, t_acc / t_numel, t_stop - t_start
        )
    )
    print("###################################")

    # Validation
    model.eval()
    valset = valset.prefetch(2)
    valiterator = dataloading.input_function(valset, config, model_input_sizes, True)
    next_element = valiterator.get_next()
    v_loss = 0
    v_acc = 0
    v_numel = 0
    v_start = time.time()
    with tf.Session() as sess:
        # for _ in range(steps):
        while True:
            try:
                x, y = sess.run(next_element)
                images = torch.Tensor(x[:, 0, :, :, :].transpose(0, 3, 1, 2))
                labels = torch.Tensor(y)

                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                log_ps = model(images)
                loss = criterion(log_ps, labels)

                # Scale the accuracy/loss to the batchsize and later divide it by number of
                # elements so unequal size batches don't scew the accuracy
                v_loss += loss.data.cpu() * images.shape[0]
                v_acc += accuracy(
                    tfmeta.metadata_.output_dim, log_ps.data.cpu(), labels.cpu(), True
                ) * images.shape[0]
                v_numel += images.shape[0]
            except tf.errors.OutOfRangeError:
                break
    v_stop = time.time()
    print("# VALIDATION #########################")
    print(
        "Loss:\t{}\nAcc:\t{}\nTime:\t{}s".format(
            t_loss / t_numel, t_acc / t_numel, t_stop - t_start
        )
    )
    print("###################################")

    return [
        t_loss / t_numel, t_acc / t_numel, (t_stop - t_start), v_loss / v_numel,
        v_acc / v_numel, (v_stop - v_start)
    ]


def testloop(model, tfdataset, model_input_sizes, output_dim, config):
    """
    # PYTORCH
    testloop uses testdata to test the pytorch model and return onehot prediciton
    values.
    """
    model.eval()
    predictions = []

    testset = tfdataset.prefetch(2)
    testiterator = dataloading.input_function(testset, config, model_input_sizes, False)
    next_element = testiterator.get_next()

    t_start = time.time()
    with tf.Session() as sess:
        # for _ in range(steps):
        while True:
            try:
                x, y = sess.run(next_element)
                images = torch.Tensor(x[:, 0, :, :, :].transpose(0, 3, 1, 2))
                # labels = torch.Tensor(y)

                if torch.cuda.is_available():
                    images = images.cuda()

                log_ps = model(images)

                probabilities = torch.sigmoid(log_ps)

                # Get a mask indicating the maximum probability for each image
                # ORIGINAL: top_p, top_class = probabilities.topk(1, dim=1)
                prediction = torch.zeros_like(probabilities, dtype=torch.uint8)
                prediction[torch.arange(len(probabilities)),
                           probabilities.argmax(dim=1)] = 1

                # Add predictions which are over a probability threshold
                if config.use_prediction_thresholding:
                    probabilities_over_threshold = probabilities > config.prediction_threshold
                    prediction = torch.max(prediction, probabilities_over_threshold)
                predictions.append(prediction.cpu().numpy())

                # Scale the accuracy/loss to the batchsize and later divide it by number of
                # elements so unequal size batches don't scew the accuracy
            except tf.errors.OutOfRangeError:
                break
    t_stop = time.time()
    print("# TEST ###############################")
    print("Time:\t{}s".format(t_stop - t_start))
    print("###################################")
    return np.concatenate(predictions)
