import os

import torchvision
import torchvision.transforms.functional as F

from PIL import Image

from src.dataset_kakaobrain import *


def get_input_size(model_name):
    if "32" in model_name:
        return 32
    elif "64" in model_name:
        return 64
    elif "128" in model_name:
        return 128
    elif "224" in model_name:
        return 224


def get_loss_criterion(classification_type):
    if classification_type == "multiclass":
        return torch.nn.CrossEntropyLoss().cuda()
    elif classification_type == "multilabel":
        return torch.nn.BCEWithLogitsLoss().cuda()
    else:
        raise ValueError("Unknown loss type")


def get_optimizer(model, optimizer_type, lr):
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr)
    else:
        raise ValueError("Unknown optimizer type")



def get_transform(is_training, input_size, scale=0.7, ratio=0.75):
    if is_training:
        return torchvision.transforms.Compose(
            [
                SelectSample(),
                AlignAxes(),
                FormatChannels(channels_des=3),
                ToPilFormat(),
                torchvision.transforms.RandomResizedCrop(
                    size=input_size, scale=(scale, 1.0), ratio=(ratio, 1 / ratio)
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                ToTorchFormat(),
            ]
        )
    else:
        return torchvision.transforms.Compose(
            [
                SelectSample(),
                AlignAxes(),
                FormatChannels(channels_des=3),
                ToPilFormat(),
                torchvision.transforms.Resize(size=(input_size, input_size)),
                ToTorchFormat(),
            ]
        )


def get_dataloader(
    model, dataset, session, is_training, first_round, batch_size, input_size, num_samples
):
    transform = get_transform(is_training=is_training, input_size=input_size)

    ds = TFDataset(session=session, dataset=dataset, num_samples=num_samples, transform=transform)

    # reduce batch size until it fits into memory
    if first_round:
        batch_size_ok = False

        while not batch_size_ok and batch_size > 1:
            ds.reset()
            try:
                dl = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, shuffle=False, drop_last=False
                )

                data, labels = next(iter(dl))
                model(data.cuda())

                batch_size_ok = True

            except RuntimeError as e:
                LOGGER.info(str(e))
                batch_size = int(batch_size / 2)
                if is_training:
                    LOGGER.info("REDUCING BATCH SIZE FOR TRAINING TO: " + str(batch_size))
                else:
                    LOGGER.info("REDUCING BATCH SIZE FOR TESTING TO: " + str(batch_size))

    LOGGER.info("USING BATCH SIZE: " + str(batch_size))
    ds.reset()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return dl, batch_size


def format_labels(labels, classification_type):
    if classification_type == "multiclass":
        return np.argmax(labels, axis=1)
    else:
        return labels


def transform_to_time_rel(t_abs):
    """
    conversion from absolute time 0s-1200s to relative time 0-1
    """
    return np.log(1 + t_abs / 60.0) / np.log(21)


def transform_to_time_abs(t_rel):
    """
    convertsion from relative time 0-1 to absolute time 0s-1200s
    """
    return 60 * (21 ** t_rel - 1)


class ToTorchFormat(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self):
        self.trans = torchvision.transforms.ToTensor()

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).float().div(255)
        else:
            # handle PIL Image
            img = self.trans(pic).float()
        return img


class SaveImage(object):
    def __init__(self, save_dir, suffix):
        self.save_dir = save_dir
        self.suffix = suffix
        self.it = 0

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            pic_temp = F.to_pil_image(pic, mode="RGB")
        elif isinstance(pic, np.ndarray):
            pic_temp = Image.fromarray(np.uint8(pic), mode="RGB")
        else:
            pic_temp = pic

        self.it += 1
        pic_temp.save(os.path.join(self.save_dir, str(self.it) + str(self.suffix) + ".jpg"))
        return pic


class Stats(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        print("shape   " + str(x.shape))
        print("min val " + str(np.array(x).min()))
        print("max val " + str(np.array(x).max()))
        return x


class SelectSample(object):
    """
    given a video 4D array, randomly select sample images within segments
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x[np.random.randint(0, x.shape[0])]


class AlignAxes(object):
    """
    Swap axes if necessary
    """

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x):
        if x.shape[0] < min(x.shape[1], x.shape[2]):
            x = np.transpose(x, (1, 2, 0))
        return x


class FormatChannels(object):
    """
    Adapt number of channels. If there are more than desired, use only the first n channels.
    If there are less, copy existing channels
    """

    def __init__(self, channels_des):
        super().__init__()
        self.channels_des = channels_des

    def __call__(self, x):
        channels = x.shape[2]
        if channels < self.channels_des:
            x = np.tile(x, (1, 1, int(np.ceil(self.channels_des / channels))))
        x = x[:, :, 0 : self.channels_des]
        return x


class ToPilFormat(object):
    """
    convert from numpy/torch array (H x W x C) to PIL images (H x W x C)
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return Image.fromarray(np.uint8(pic * 255), mode="RGB")
        elif isinstance(pic, torch.Tensor):
            return F.to_pil_image(pic)
        else:
            raise TypeError("unknown input type")


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, class_index):
        with open(data_path, "rb") as fh:
            self.dataset = torch.tensor(pickle.load(fh)).float()
            self.class_index = torch.tensor(class_index).float()

    def get_dataset(self):
        # for compatibility
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.class_index


def load_datasets_processed(cfg, datasets, dataset_dir=None):
    """
    load preprocessed datasets from a list, return train/test datasets, dataset index and dataset name
    """
    if dataset_dir is None:
        dataset_dir = cfg["proc_dataset_dir"]
    dataset_list = []
    class_index = 0

    for dataset_name in datasets:
        dataset_train_path = os.path.join(dataset_dir, dataset_name + "_train")
        dataset_test_path = os.path.join(dataset_dir, dataset_name + "_test")

        try:
            dataset_train = ProcessedDataset(dataset_train_path, class_index)
            dataset_test = ProcessedDataset(dataset_test_path, class_index)
        except Exception as e:
            print(e)
            continue

        dataset_list.append((dataset_train, dataset_test, dataset_name, class_index))
        class_index += 1

    return dataset_list
