# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon
"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

# fmt: off
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath("."))  # This is needed for the run_local_test ingestion

import yaml
from src.architectures.architectures import *
from src.utils import *

# fmt: on

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(
        self, metadata, model_config=None, model_dir="/home/ferreira/autodl_data/models_thomas"
    ):
        LOGGER.info("INIT START: " + str(time.time()))
        super().__init__()

        self.time_start = time.time()

        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()

        self.train_round = 0  # flag indicating if we are in the first round of training
        self.test_round = 0
        self.train_counter = 0
        self.train_batches = 0
        self.dl_train = None
        self.classification_type = None

        # hyperparameters
        self.model_dir = model_dir

        with open(model_config) as stream:
            self.model_config = yaml.safe_load(stream)

        print(self.model_config)
        self.batch_size_test = 512

        self.session = tf.Session()
        LOGGER.info("INIT END: " + str(time.time()))

    def train(self, dataset, remaining_time_budget=None):
        desired_batches = 100  # Hard coded.. originally computed from timings.pkl from thomas

        LOGGER.info("TRAINING START: " + str(time.time()))
        LOGGER.info("NUM SAMPLES: " + str(desired_batches))

        self.train_round += 1

        # initial config during first round
        if int(self.train_round) == 1:
            self.late_init(dataset)

        torch.set_grad_enabled(True)
        self.model.train()
        self.model.eval()
        self.model.train()

        finish_loop = False

        if self.train_batches >= desired_batches:
            return self.train_batches

        LOGGER.info("TRAIN BATCH START")
        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for i, (data, labels) in enumerate(self.dl_train):
                self.optimizer.zero_grad()
                output = self.model(data.cuda())
                labels = format_labels(labels, self.classification_type).cuda()
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.train_counter += self.batch_size_train
                self.train_batches += 1

                LOGGER.info("TRAIN: " + str(i))

                if self.train_batches > desired_batches:
                    finish_loop = True
                    break

            LOGGER.info("TRAIN COUNTER: " + str(self.train_counter))
            self.train_round += 1

        LOGGER.info("TRAIN BATCH END")
        LOGGER.info("LR: ")
        for param_group in self.optimizer.param_groups:
            LOGGER.info(param_group["lr"])
        LOGGER.info(
            "TRAINING FRAMES PER SEC: " + str(self.train_counter / (time.time() - self.time_start))
        )
        LOGGER.info("TRAINING COUNTER: " + str(self.train_counter))
        LOGGER.info("TRAINING BATCHES: " + str(self.train_batches))
        LOGGER.info("TRAINING END: " + str(time.time()))

        return self.train_batches

    def late_init(self, dataset):
        LOGGER.info("INIT")

        ds_temp = TFDataset(session=self.session, dataset=dataset)
        self.info = ds_temp.scan(25)

        LOGGER.info("AVG SHAPE: " + str(self.info["avg_shape"]))

        if self.info["is_multilabel"]:
            self.classification_type = "multilabel"
        else:
            self.classification_type = "multiclass"

        print(type(self.model_config["model"]))

        self.model = get_model(
            model_name=self.model_config["model"],
            model_dir=self.model_dir,
            dropout=self.model_config["dropout"],
            num_classes=self.num_classes,
        )
        self.input_size = get_input_size(self.model_config["model"])
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_type=self.model_config["optimizer"],
            lr=self.model_config["lr"]
        )
        self.criterion = get_loss_criterion(classification_type=self.classification_type)

        self.dl_train, self.batch_size_train = get_dataloader(
            model=self.model,
            dataset=dataset,
            session=self.session,
            is_training=True,
            first_round=(int(self.train_round) == 1),
            batch_size=self.model_config["batch_size_train"],
            input_size=self.input_size,
            num_samples=int(10000000),
        )

    def test(self, dataset, remaining_time_budget=None):
        LOGGER.info("TESTING START: " + str(time.time()))

        self.test_round += 1

        if int(self.test_round) == 1:
            scan_start = time.time()
            ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=10000000)
            info = ds_temp.scan()
            self.num_samples_test = info["num_samples"]
            LOGGER.info("SCAN TIME: " + str(time.time() - scan_start))
            LOGGER.info("TESTING: FIRST ROUND")

        torch.set_grad_enabled(False)
        self.model.eval()
        dl, self.batch_size_test = get_dataloader(
            model=self.model,
            dataset=dataset,
            session=self.session,
            is_training=False,
            first_round=(int(self.test_round) == 1),
            batch_size=self.batch_size_test,
            input_size=self.input_size,
            num_samples=self.num_samples_test,
        )

        LOGGER.info("TEST BATCH START")
        prediction_list = []
        for i, (data, _) in enumerate(dl):
            LOGGER.info("TEST: " + str(i))
            prediction_list.append(self.model(data.cuda()).cpu())
        predictions = np.vstack(prediction_list)
        LOGGER.info("TEST BATCH END")

        LOGGER.info("TESTING END: " + str(time.time()))
        return predictions
