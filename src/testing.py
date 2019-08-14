import time
import numpy as np
import torch
from utils import LOGGER, DEVICE


class default_tester():
    def __init__(self, num_segments_test, bn_prod_limit):
        self.num_segments_test = num_segments_test
        self.bn_prod_limit = bn_prod_limit
        self.test_time = 0

    def __call__(self, autodl_model, remaining_time):
        predictions = []
        if (
            autodl_model.test_dl.dataset.max_shape[0] > 1
            and self.num_segments_test is not None
        ):
            autodl_model.model.num_segments = self.num_segments_test

        self.update_batch_size(autodl_model)
        LOGGER.info('NUM_SEGMENTS: ' + str(autodl_model.model.num_segments))
        LOGGER.info('LR: {0:.4e}'.format(autodl_model.optimizer.param_groups[0]['lr']))
        LOGGER.info('DROPOUT: {0:.4g}'.format(autodl_model.model.dropout))

        with torch.no_grad():
            test_start = time.time()
            autodl_model.model.eval()
            autodl_model.test_dl.dataset.reset()
            for i, (data, _) in enumerate(autodl_model.test_dl):
                LOGGER.debug('TEST BATCH #' + str(i))
                data = data.to(DEVICE)
                output = autodl_model.model(data)
                predictions += output.cpu().tolist()
                i += 1

        autodl_model.test_time.append(time.time() - test_start)
        return np.array(predictions)

    def update_batch_size(self, autodl_model):
        batch_size = int(self.bn_prod_limit / autodl_model.model.num_segments)
        if batch_size == autodl_model.test_dl.batch_size:
            return
        trainloader_args = {**autodl_model.config.dataloader_args['test']}
        trainloader_args['batch_size'] = batch_size
        autodl_model.test_dl = torch.utils.data.DataLoader(
            autodl_model.test_dl.dataset,
            **trainloader_args
        )
        LOGGER.info('BATCH SIZE: ' + str(autodl_model.test_dl.batch_size))
