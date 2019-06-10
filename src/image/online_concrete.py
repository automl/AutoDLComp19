import torch
import dataloading
import numpy as np
import torch.nn as nn


def trainloop(model, unfrozen_parameters, train_data_iterator, config, steps):
    """
    # PYTORCH
    Trainloop function does the actual training of the model
    1) it gets the X, y from tensorflow dataset.
    2) convert X, y to CUDA
    3) trains the model with the Tesors for given no of steps.
    """
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for i in range(steps):
        images, labels = dataloading.get_torch_tensors(train_data_iterator)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)

        if torch.cuda.is_available():
            images = images.float().cuda()
            labels = labels.long().cuda()
        else:
            images = images.float()
            labels = labels.long()
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()


def testloop(model, dataloader, output_dim):
    """
    # PYTORCH
    testloop uses testdata to test the pytorch model and return onehot prediciton
    values.
    """
    preds = []
    with torch.no_grad():
        model.eval()
        for [images] in dataloader:
            if torch.cuda.is_available():
                images = images.float().cuda()
            else:
                images = images.float()
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            preds.append(top_class.cpu().numpy())
    preds = np.concatenate(preds)
    onehot_preds = np.squeeze(np.eye(output_dim)[preds.reshape(-1)])
    return onehot_preds
