import time

import numpy as np
import torch
from utils import DEVICE, LOGGER

class DefaultPredictor():
    def __init__(self, **tester_args):
        self.test_time = 0
        self.entropy_idx = None
        self.test_iteration = 0
        self.predictions = None
        self.tester_args = tester_args
        print('tester_args')
        print(self.tester_args)

    def __call__(self, autodl_model, remaining_time):
        '''
        This is called from the model.py and just seperates the
        testing routine from the unchaning code
        '''
        if self.test_iteration == 0 or self.test_iteration > int(self.tester_args['entropy_splits']):
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
                    output = torch.sigmoid(output)
                    predictions += output.cpu().tolist()
                    i += 1

                print('????????? time 1 ????????? ' + str(time.time()-test_start))

            autodl_model.test_time.append(time.time() - test_start)
            entropy = self.calculate_entropy(np.array(predictions))
            self.entropy_idx = np.flip(np.argsort(entropy))
            self.predictions = np.array(predictions)
        else:
            lng = len(self.entropy_idx)
            idx_start = int(lng*(self.test_iteration-1)/self.tester_args['entropy_splits'])
            idx_end   = int(lng*(self.test_iteration)/self.tester_args['entropy_splits'])
            entropy_idx = np.sort(self.entropy_idx[idx_start:idx_end])

            predictions = []
            with torch.no_grad():
                test_start = time.time()
                #######################################################################
                ## NOTE(Philipp): Never set the train/eval mode here                 ##
                ## IT SHOULD BE SET IN THE POLICY WHEN DECIDING TO MAKE A PREDICTION ##
                #######################################################################

                # Just making sure we start at the beginning
                autodl_model.test_dl.dataset.reset()
                autodl_model.test_dl.set_entropy_idx(entropy_idx)

                for i, (data, _) in enumerate(autodl_model.test_dl):
                    LOGGER.debug('TEST BATCH #{}'.format(i))
                    data = data.to(DEVICE)
                    output = autodl_model.model(data)
                    output = torch.sigmoid(output)
                    predictions += output.cpu().tolist()
                    i += 1

                print('????????? time 2 ????????? ' + str(time.time() - test_start))

            autodl_model.test_time.append(time.time() - test_start)
            predictions = np.array(predictions)
            self.predictions[entropy_idx,:] = predictions

        print('????????? test iteration ????????? ' + str(self.test_iteration))

        self.test_iteration +=1
        return self.predictions



    def calculate_entropy(self, predictions):
        return np.sum(np.log(predictions),axis=1)*(-1.0 / predictions.shape[1])

