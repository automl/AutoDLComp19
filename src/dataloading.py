import numpy as np
import tensorflow as tf
import torch
import torch.utils.data as data_utils
import utils


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
            "Detected that examples have variable sequence_size, will " +
            "randomly crop a sequence with num_frames = " + "{}".format(num_frames)
        )
        tensor_4d = _crop_time_axis(tensor_4d, num_frames=num_frames)

    # Row and col
    new_row_count = image_size[0]
    new_col_count = image_size[1]

    utils.print_log(
        "Will resize space axes to (new_row_count, new_col_count) = " +
        "{}".format((new_row_count, new_col_count))
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


def input_function(dataset, config, image_size, is_training):
    """Given `dataset` received by the method `self.train` or `self.test`,
    prepare input to feed to model function.

    For more information on how to write an input function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function

    # PYTORCH
    This function returns a tensorflow data iterator which is then converted to
    PyTorch Tensors.
    """

    dataset = dataset.map(
        lambda *x: (_preprocess_tensor_4d(x[0], config, image_size), x[1])
    )

    if is_training:
        # Shuffle input examples
        dataset = dataset.shuffle(buffer_size=config.default_shuffle_buffer)
        # Convert to RepeatDataset to train for several epochs
        dataset = dataset.repeat()

    # Set batch size
    dataset = dataset.batch(batch_size=config.batch_size)

    return dataset.make_one_shot_iterator()
