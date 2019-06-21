import dataloading
import numpy as np
import torch
import torch.nn as nn
from apex import amp


def trainloop(model, unfrozen_parameters, train_data_iterator, config, steps):
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
    optimizer = torch.optim.Adam(unfrozen_parameters, lr=config.lr)

    # Mixed precision monkey patch
    model, optimizer = amp.initialize(model, optimizer, opt_level=config.mixed_precision)

    for i in range(steps):
        images, labels = dataloading.get_torch_tensors(train_data_iterator)
        images = torch.Tensor(images).float()
        labels = torch.Tensor(labels).float()

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()


def testloop(model, dataloader, output_dim, config):
    """
    # PYTORCH
    testloop uses testdata to test the pytorch model and return onehot prediciton
    values.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for [images] in dataloader:
            images = images.float()
            if torch.cuda.is_available():
                images = images.cuda()

            probabilities = torch.sigmoid(model(images))

            # Get a mask indicating the maximum probability for each image
            # ORIGINAL: top_p, top_class = probabilities.topk(1, dim=1)
            prediction = torch.zeros_like(probabilities, dtype=torch.uint8)
            prediction[torch.arange(len(probabilities)), probabilities.argmax(dim=1)] = 1

            # Add predictions which are over a probability threshold
            if config.use_prediction_thresholding:
                probabilities_over_threshold = probabilities > config.prediction_threshold
                prediction = torch.max(prediction, probabilities_over_threshold)
            predictions.append(prediction.cpu().numpy())

    # Flatten over batches as scoring.py is batch agnostic
    # ORIGINAL: return np.squeeze(np.eye(output_dim)[predictions.reshape(-1)])
    return np.concatenate(predictions)
