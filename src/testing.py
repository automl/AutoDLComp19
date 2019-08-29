import time

import numpy as np
import psutil
import torch
from utils import DEVICE, LOGGER


class DefaultPredictor():
    '''
    The default predictor will cache the test dataset if enough memory
    is available to speed up making predictions after the first
    prediction round
    '''
    def __init__(self, use_cache=True):
        self.test_time = 0
        self.cache = None
        self.use_cache = None if use_cache else False

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        testing routine from the unchaning code
        '''
        predictions = []
        temp_cache = None
        num_samples = autodl_model.test_num_samples
        batch_size = autodl_model.test_dl.batch_size
        with torch.no_grad():
            test_start = time.time()
            #######################################################################
            ## NOTE(Philipp): Never set the train/eval mode here                 ##
            ## IT SHOULD BE SET IN THE POLICY WHEN DECIDING TO MAKE A PREDICTION ##
            #######################################################################

            # Just making sure we start at the beginning
            autodl_model.test_dl.dataset.reset()
            e = enumerate(autodl_model.test_dl) if self.cache is None else enumerate(
                zip(
                    chunk(self.cache, batch_size),
                    range(int(np.ceil(num_samples / batch_size)))
                )
            )
            for i, (data, _) in e:
                LOGGER.debug('TEST BATCH #{}'.format(i))
                cudata = data.to(DEVICE)
                output = autodl_model.model(cudata)
                predictions += output.cpu().tolist()
                i += 1
                if self.use_cache is None:
                    ele_mem_size = (data.element_size() * data.nelement()) / batch_size
                    available_mem = psutil.virtual_memory().available
                    max_count = available_mem / ele_mem_size
                    # Inflate estimated ram-usage by 10% to not hog all memory available
                    self.use_cache = max_count > num_samples * 1.1
                if self.use_cache and self.cache is None:
                    if temp_cache is None:
                        # Preallocate space for the data
                        temp_cache = torch.full(
                            (num_samples, *data.size()[1:]),
                            float('nan'),
                            dtype=data.dtype,
                            pin_memory=True
                        )
                    si = (i - 1) * batch_size
                    temp_cache[si:si + data.size()[0]] = data

        if self.use_cache and self.cache is None:
            # assert(not torch.any(torch.isnan(temp_cache)))
            self.cache = temp_cache
        autodl_model.test_time.append(time.time() - test_start)
        return np.array(predictions)


def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
