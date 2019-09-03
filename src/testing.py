import time

import numpy as np
import psutil
import torch
from utils import DEVICE, GB, LOGGER, MB, memprofile


class DefaultPredictor():
    '''
    The default predictor will cache the test dataset if enough memory
    is available to speed up making predictions after the first
    prediction round
    '''
    def __init__(self, use_cache=True):
        self.test_time = 0
        self.test_cache = None
        self.use_cache = None if use_cache else False

    @memprofile(precision=2)
    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just separates the
        testing routine from the unchanging code
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
            autodl_model.test_dl.reset()
            e = enumerate(autodl_model.test_dl) if self.test_cache is None else enumerate(
                zip(
                    chunk(self.test_cache, batch_size),
                    range(int(np.ceil(num_samples / batch_size)))
                )
            )
            batch_loading_time = 0
            i = -1
            load_start = time.time()
            for i, (data, _) in e:
                batch_loading_time += time.time() - load_start

                LOGGER.debug('TEST BATCH #{}'.format(i))
                cudata = data.to(DEVICE, non_blocking=True)
                output = autodl_model.model(cudata)
                predictions += output.cpu().tolist()
                i += 1
                if self.use_cache is None:
                    ele_mem_size = (data.element_size() * data.nelement()) / batch_size
                    # Inflate estimated ram-usage by 1 GB to not hog all memory available
                    available_mem = psutil.virtual_memory().available - 1 * GB
                    max_count = available_mem / ele_mem_size
                    self.use_cache = max_count > num_samples * 1.1
                if self.use_cache and self.test_cache is None:
                    if temp_cache is None:
                        # Preallocate space for the data
                        temp_cache = torch.empty(
                            (num_samples, *data.size()[1:]),
                            dtype=data.dtype,
                            pin_memory=True
                        )
                        LOGGER.debug(
                            'ALLOCATED {0:.2f} MB TO CACHE TEST DATA'.format(
                                temp_cache.element_size() * temp_cache.nelement() / MB
                            )
                        )
                    si = (i - 1) * batch_size
                    temp_cache[si:si + data.size()[0]] = data
                load_start = time.time()
            if i >= 0:
                i += 1
                LOGGER.debug(
                    'SEC PER BATCH LOADING:\t{0:.4f}'.format(batch_loading_time / i)
                )
        if self.use_cache and self.test_cache is None:
            # assert(not torch.any(torch.isnan(temp_cache)))
            self.test_cache = temp_cache
        autodl_model.test_time.append(time.time() - test_start)
        return np.array(predictions)


def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
