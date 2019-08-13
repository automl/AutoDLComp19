import time
import numpy as np
import torch
from utils import LOGGER


class default_tester():
    def __init__(self, num_segments_test):
        self.num_segments_test = num_segments_test
        self.test_time = 0

    def __call__(self, autodl_model, remaining_time):
        predictions = []
        if autodl_model.test_dl.dataset.max_shape[0] > 1:
            autodl_model.model.num_segments = self.num_segments_test
        test_start = time.time()
        autodl_model.test_dl.dataset.reset()
        with torch.no_grad():
            for i, (data, _) in enumerate(autodl_model.test_dl):
                autodl_model.model.train()
                LOGGER.info('test: ' + str(i))
                data = data.cuda()
                output = autodl_model.model(data)
                predictions += output.cpu().tolist()
                i += 1
        autodl_model.test_time.append(time.time() - test_start)
        return np.array(predictions)
