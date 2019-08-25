import time

import numpy as np
import torch
from utils import DEVICE, LOGGER


class DefaultPredictor():
    def __init__(self):
        self.test_time = 0

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        testing routine from the unchaning code
        '''
        predictions = []
        with torch.no_grad():
            test_start = time.time()
            #######################################################################
            ## NOTE(Philipp): Never set the train/eval mode here                 ##
            ## IT SHOULD BE SET IN THE POLICY WHEN DECIDING TO MAKE A PREDICTION ##
            #######################################################################

            # Just making sure we start at the beginning
            autodl_model.test_dl.dataset.reset()
            for i, (data, _) in enumerate(autodl_model.test_dl):
                LOGGER.debug('TEST BATCH #{}'.format(i))
                data = data.to(DEVICE)
                output = autodl_model.model(data)
                predictions += output.cpu().tolist()
                i += 1

        autodl_model.test_time.append(time.time() - test_start)
        return np.array(predictions)
