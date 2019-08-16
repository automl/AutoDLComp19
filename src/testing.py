import time
import numpy as np
import torch
import torch.cuda as cutorch
from torch_adapter import TFDataLoader
from utils import LOGGER, DEVICE


class baseline_tester():
    def __init__(self, never_leave_train_mode):
        self.never_leave_train_mode = never_leave_train_mode
        self.test_time = 0

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        testing routine from the unchaning code
        '''
        predictions = []
        vram_cached = []

        LOGGER.info('NUM_SEGMENTS: ' + str(autodl_model.model.num_segments))
        LOGGER.info('LR: {0:.4e}'.format(autodl_model.optimizer.param_groups[0]['lr']))
        LOGGER.info('DROPOUT: {0:.4g}'.format(autodl_model.model.dropout))

        with torch.no_grad():
            test_start = time.time()
            if self.never_leave_train_mode:
                LOGGER.debug('##########################################################')
                LOGGER.debug('MODEL IS IN TRAIN MODE')
                has_frozen = np.any([not m.training for m in autodl_model.model.modules()])
                LOGGER.debug('MODEL HAS FROZEN MODULES:\t{0}'.format(has_frozen))
                LOGGER.debug('##########################################################')
                autodl_model.model._modules['baseline_aug_net'].eval()
                LOGGER.debug('MODEL IS IN EVAL MODE')
                has_unfrozen = np.any([m.training for m in autodl_model.model.modules()])
                LOGGER.debug('MODEL HAS UNFROZEN MODULES:\t{0}'.format(has_unfrozen))
                LOGGER.debug('##########################################################')
                LOGGER.debug('All done. You can get back to work now!')
                LOGGER.debug('##########################################################')
            else:
                autodl_model.model.eval()

            test_finished = False
            autodl_model.test_dl.dataset.reset()  # Just making sure we start at the beginning
            while not test_finished:
                try:
                    for i, (data, _) in enumerate(autodl_model.test_dl):
                        LOGGER.debug('TEST BATCH #' + str(i))
                        data = data.to(DEVICE)
                        output = autodl_model.model(data)
                        predictions += output.cpu().tolist()
                        i += 1

                        vram_cached.append(cutorch.memory_cached())
                    test_finished = True
                except RuntimeError as e:
                    # If we are out of vram reduce the batchsize by factor 2 and try again
                    if 'CUDA out of memory.' not in e.args[0]:
                        raise e
                    predictions = []
                    vram_cached = []

                    loader_args = autodl_model.config.dataloader_args['test']
                    loader_args.update({'batch_size': max(16, int(autodl_model.test_dl.batch_size - 25))})
                    autodl_model.test_dl = TFDataLoader(
                        autodl_model.test_dl.dataset,
                        **loader_args
                    )
                    autodl_model.test_dl.dataset.reset()
                    LOGGER.info('BATCH SIZE CHANGED: {}'.format(autodl_model.test_dl.batch_size))

        LOGGER.info('AVERAGE VRAM USAGE: {0:.2f} MB'.format(np.mean(vram_cached) / 1024**2))
        autodl_model.test_time.append(time.time() - test_start)
        return np.array(predictions)


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
