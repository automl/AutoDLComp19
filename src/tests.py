import time

import numpy as np
import tensorflow as tf
import torch
from torch_adapter import TFDataLoader, TFDataset
from utils import DEVICE, LOGGER


def _benchmark_loading_and_transformations(autodl_model, ds_temp):
    import transformations.video

    def test_pipelinespeed(ds_temp, max_i, model):
        dl_loadtime = 0
        numel = 1e-12
        t_s = time.time()
        for i, (d, l) in enumerate(ds_temp):
            if type(d) is not torch.Tensor:
                d = torch.Tensor(d).pin_memory()
            dl_loadtime += time.time() - t_s
            numel += len(d)

            d = d.to(DEVICE, non_blocking=True)
            model(d)
            if i > max_i:
                break
            t_s = time.time()
        LOGGER.debug('{0:.6f} s/d'.format(dl_loadtime / numel))

    # Test chosen transformation against...
    LOGGER.debug(50 * '#')
    get_and_apply_transformations = getattr(transformations.video, 'normal_segment_dist')
    model, transf = get_and_apply_transformations(autodl_model.model.main_net, ds_temp)
    model.to(DEVICE, non_blocking=True)

    dataset = autodl_model._tf_train_set.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    ds_temp = TFDataset(
        autodl_model.session, dataset, autodl_model.train_num_samples,
        transf['train']['samples'], transf['train']['labels']
    )
    dl_temp = TFDataLoader(ds_temp, 16)

    dl_temp.dataset.reset()
    test_pipelinespeed(dl_temp, 60, model)
    dl_temp.dataset.reset()
    test_pipelinespeed(dl_temp, 60, model)
    dl_temp.dataset.reset()
    test_pipelinespeed(dl_temp, 60, model)

    # Another transformation stack
    LOGGER.debug(50 * '#')
    get_and_apply_transformations = getattr(
        transformations.video, 'resize_normal_seg_selection'
    )
    model, transf = get_and_apply_transformations(
        autodl_model.model.main_net, ds_temp, 0.7
    )
    model.to(DEVICE, non_blocking=True)

    dataset = autodl_model._tf_train_set.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    ds_temp = TFDataset(
        autodl_model.session, dataset, autodl_model.train_num_samples,
        transf['train']['samples'], transf['train']['labels']
    )
    dl_temp = TFDataLoader(ds_temp, 16)

    dl_temp.dataset.reset()
    test_pipelinespeed(dl_temp, 60, model)
    dl_temp.dataset.reset()
    test_pipelinespeed(dl_temp, 60, model)
    dl_temp.dataset.reset()
    test_pipelinespeed(dl_temp, 60, model)

    # Using the transformation stack's cpu part with the tf pipeline
    LOGGER.debug(50 * '#')
    get_and_apply_transformations = getattr(transformations.video, 'normal_segment_dist')
    model, transf = get_and_apply_transformations(autodl_model.model.main_net, ds_temp)
    model.to(DEVICE, non_blocking=True)

    def trans(x, y):
        ret = (
            transf['train']['samples'](x),
            transf['train']['labels'](y),
        )
        return ret

    def tfwrap(x, y):
        ret = tf.py_func(trans, [x, y], [tf.float32, tf.int64])
        return ret

    dataset = autodl_model._tf_train_set.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(tfwrap, num_parallel_calls=10)
    dataset = dataset.batch(16)
    ds_temp = TFDataset(autodl_model.session, dataset, autodl_model.train_num_samples)

    ds_temp.reset()
    test_pipelinespeed(ds_temp, 60, model)
    ds_temp.reset()
    test_pipelinespeed(ds_temp, 60, model)
    ds_temp.reset()
    test_pipelinespeed(ds_temp, 60, model)

    # Using the transformation stack's cpu part with the tf pipeline
    LOGGER.debug(50 * '#')
    get_and_apply_transformations = getattr(
        transformations.video, 'resize_normal_seg_selection'
    )
    model, transf = get_and_apply_transformations(
        autodl_model.model.main_net, ds_temp, 0.7
    )
    model.to(DEVICE, non_blocking=True)

    def trans2(x, y):
        ret = (
            transf['train']['samples'](x),
            transf['train']['labels'](y),
        )
        return ret

    def tfwrap2(x, y):
        ret = tf.py_func(trans2, [x, y], [tf.float32, tf.int64])
        return ret

    dataset = autodl_model._tf_train_set.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(tfwrap2, num_parallel_calls=10)
    dataset = dataset.batch(16)
    ds_temp = TFDataset(autodl_model.session, dataset, autodl_model.train_num_samples)

    ds_temp.reset()
    test_pipelinespeed(ds_temp, 60, model)
    ds_temp.reset()
    test_pipelinespeed(ds_temp, 60, model)
    ds_temp.reset()
    test_pipelinespeed(ds_temp, 60, model)

    # Compare split speeds - this is stupid
    # skip induces latency and shard probably
    # just spreads these skip across the dataset I assume
    LOGGER.debug(50 * '#')
    get_and_apply_transformations = getattr(
        transformations.video, 'resize_normal_seg_selection'
    )
    model, transf = get_and_apply_transformations(
        autodl_model.model.main_net, ds_temp, 0.7
    )
    model.to(DEVICE, non_blocking=True)

    def trans3(x, y):
        ret = (
            transf['train']['samples'](x),
            transf['train']['labels'](y),
        )
        return ret

    def tfwrap3(x, y):
        ret = tf.py_func(trans3, [x, y], [tf.float32, tf.int64])
        return ret

    ds1 = autodl_model._tf_train_set.take(int(autodl_model.train_num_samples * 0.5))

    ds1 = ds1.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds1 = ds1.map(tfwrap3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds1 = ds1.batch(16)
    ds1_temp = TFDataset(autodl_model.session, ds1, autodl_model.train_num_samples)

    ds1_temp.reset()
    test_pipelinespeed(ds1_temp, 60, model)
    ds1_temp.reset()
    test_pipelinespeed(ds1_temp, 60, model)
    ds1_temp.reset()
    test_pipelinespeed(ds1_temp, 60, model)

    ds2 = ds1.shard(10, 9)
    ds2_temp = TFDataset(autodl_model.session, ds2, autodl_model.train_num_samples)

    ds2_temp.reset()
    test_pipelinespeed(ds2_temp, 60, model)
    ds2_temp.reset()
    test_pipelinespeed(ds2_temp, 60, model)
    ds2_temp.reset()
    test_pipelinespeed(ds2_temp, 60, model)


def _check_for_shuffling(autodl_model, ds_temp):
    LOGGER.debug('SANITY CHECKING SHUFFLING')
    ds_train = TFDataset(
        session=autodl_model.session,
        dataset=ds_temp.dataset,
        num_samples=min(ds_temp.num_samples, 100)
    )
    ds_train.reset()
    ds_temp.reset()
    dset1 = [e for e in ds_train]  # Check if next time around
    dset2 = [e for e in ds_train]  # the order is shuffled
    dset3 = [e for e in ds_temp][:autodl_model.train_num_samples]
    i = 0
    e1vse2 = []
    e2vse3 = []
    for e1, e2, e3 in zip(dset1, dset2, dset3):
        if i % 100 == 0:
            LOGGER.debug('Checking i: {}'.format(i))
        e1vse2.append((np.all((e1[0] == e2[0]))))
        e2vse3.append((np.all((e2[0] == e3[0]))))
        i += 1
    LOGGER.debug('E1 == E2: {}\t should be False'.format(np.all(e1vse2)))
    LOGGER.debug('E2 == E3: {}\t should be False'.format(np.all(e2vse3)))
