import tensorflow as tf
import utils
import numpy as np
import torch.utils.data as data_utils
import torch


def _crop_time_axis(tensor_4d, num_frames, begin_index=None):
    """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.
    Args:
        tensor_4d: A Tensor of shape
            [sequence_size, row_count, col_count, num_channels]
        num_frames: An integer representing the resulted chunk (sequence) length
        begin_index: The index of the beginning of the chunk. If `None`, chosen
          randomly.
    Returns:
        A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
    """
    # pad sequence if not long enough
    pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[1], 0)
    padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

    # If not given, randomly choose the beginning index of frames
    if not begin_index:
        maxval = tf.shape(padded_tensor)[0] - num_frames + 1
        begin_index = tf.random.uniform([1], minval=0, maxval=maxval, dtype=tf.int32)
        begin_index = tf.stack([begin_index[0], 0, 0, 0], name="begin_index")

    sliced_tensor = tf.slice(
        padded_tensor, begin=begin_index, size=[num_frames, -1, -1, -1]
    )

    return sliced_tensor


def _resize_space_axes(tensor_4d, new_row_count, new_col_count):
    """Given a 4-D tensor, resize space axes to have target size.
    Args:
        tensor_4d: A Tensor of shape
            [sequence_size, row_count, col_count, num_channels].
        new_row_count: An integer indicating the target row count.
        new_col_count: An integer indicating the target column count.
    Returns:
        A Tensor of shape [sequence_size, target_row_count, target_col_count].
    """
    resized_images = tf.image.resize_images(
        tensor_4d, size=(new_row_count, new_col_count)
    )
    return resized_images


def _preprocess_tensor_4d(tensor_4d, config, image_size):
    """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
    non-fixed). The output tensor wil have fixed, known shape.
    Args:
        tensor_4d: A Tensor of shape
            [sequence_size, row_count, col_count, num_channels]
            where some dimensions might be `None`.
    Returns:
        A 4-D Tensor with fixed, known shape.
    """
    tensor_4d_shape = tensor_4d.shape
    utils.print_log("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

    # Frames
    if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
        num_frames = tensor_4d_shape[0]
    else:
        num_frames = config.default_num_frames

    if not tensor_4d_shape[0] > 0:
        utils.print_log(
            "Detected that examples have variable sequence_size, will "
            + "randomly crop a sequence with num_frames = "
            + "{}".format(num_frames)
        )
        tensor_4d = _crop_time_axis(tensor_4d, num_frames=num_frames)

    # Row and col
    new_row_count = image_size[0]
    new_col_count = image_size[1]

    utils.print_log(
        "Will resize space axes to (new_row_count, new_col_count) = "
        + "{}".format((new_row_count, new_col_count))
    )
    tensor_4d = _resize_space_axes(
        tensor_4d, new_row_count=new_row_count, new_col_count=new_col_count
    )

    # Channels
    if tensor_4d_shape[3] == 1:
        # TODO(Danny): Is this inefficient? Should this be done here?
        tensor_4d = tf.image.grayscale_to_rgb(tensor_4d)

    utils.print_log("Tensor shape after preprocessing: {}".format(tensor_4d.shape))
    return tensor_4d


def get_dataloader(tfdataset, config, image_size, train=False):
    """
    # PYTORCH
    This function takes a tensorflow dataset class and comvert it into a
    Pytorch Dataloader of batchsize.
    This function is usually used for testing data alone, as training data
    is huge and training is done step/batch wise, rather than epochs.
    """
    tfdataset = tfdataset.map(
        lambda *x: (_preprocess_tensor_4d(x[0], config, image_size), x[1])
    )
    iterator = tfdataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    features = []
    labels = []
    if train:
        while True:
            try:
                x, y = sess.run(next_element)
                x = x.transpose(0, 3, 1, 2)
                y = y.argmax()
                features.append(x)
                labels.append(y)
            except tf.errors.OutOfRangeError:
                break
        features = np.vstack(features)
        features = torch.Tensor(features)
        labels = torch.Tensor(labels)
        dataset = data_utils.TensorDataset(features, labels)
        loader = data_utils.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
    else:
        while True:
            try:
                x, _ = sess.run(next_element)
                x = x.transpose(0, 3, 1, 2)
                features.append(x)
            except tf.errors.OutOfRangeError:
                break
        features = np.vstack(features)
        features = torch.Tensor(features)
        dataset = data_utils.TensorDataset(features)
        loader = data_utils.DataLoader(dataset, batch_size=config.batch_size)
    return loader


def get_torch_tensors(training_data_iterator):
    """
    # PYTORCH
    This function returns X and y Torch tensors from the tensorflow
    data iterator.
    X is transposed as images need specific dimensions in PyTorch.
    y is converted to single integer value from One-hot vectors.
    """
    next_iter = training_data_iterator.get_next()
    sess = tf.Session()
    images, labels = sess.run(next_iter)
    return images[:, 0, :, :, :].transpose(0, 3, 1, 2), np.argmax(labels, axis=1)


def input_function(dataset, config, list_of_transforms, is_training, is_validation, num_epochs, train_size=None,
                   default_shuffle_buffer=None):
    """
    TensorFlow dataset iterator for AutoDL dataset.
    :param dataset: TensorFlow dataset
    :param config: local config file
    :param list_of_transforms: List of functions to use for preprocessing in the order given. For an example on how to write them please have a look at https://www.tensorflow.org/guide/datasets#decoding_image_data_and_resizing_it
    :param is_training: Training phase?
    :param is_validation: During training are we validating?
    :param num_epochs: Number of epochs to train
    :param train_size:
    :param default_shuffle_buffer: size of the shuffle buffer
    :return: tf data iterator
    """
    if train_size is not None and is_training:
        if is_validation:
            # Skip the training elements in the dataset
            # to get to the validation
            dataset = dataset.skip(train_size)
        else:
            # Only keep the elements in training
            dataset = dataset.take(train_size)

    if not default_shuffle_buffer:
        default_shuffle_buffer = config.default_shuffle_buffer

    for transform_function in list_of_transforms:
        dataset = dataset.map(transform_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        # Shuffle input examples
        dataset = dataset.shuffle(buffer_size=default_shuffle_buffer)
        # Convert to RepeatDataset to train for several epochs
        dataset = dataset.repeat(num_epochs)

    # Create batches
    dataset = dataset.batch(batch_size=config.batch_size)
    # Prefetch next batches already
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset.make_one_shot_iterator()
