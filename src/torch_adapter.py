import multiprocessing
import queue
import threading

import numpy as np
import tensorflow as tf
import torch
import torch.utils.data._utils as _utils
from torch.utils.data import BatchSampler, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate


class TFDataset(Dataset):
    def __init__(
        self,
        session,
        dataset,
        num_samples=None,
        transform_sample=None,
        transform_label=None
    ):
        super(TFDataset, self).__init__()
        self.session = session
        self.dataset = dataset
        self.transform_sample = transform_sample
        self.transform_label = transform_label

        # Metadata
        self.num_samples = num_samples
        self.num_classes = None
        self.min_shape = None
        self.max_shape = None
        self.median_shape = None
        self.mean_shape = None
        self.std_shape = None
        self.is_multilabel = None

        self.current_idx = 0

        self.next_element = None
        self.reset()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        try:
            example, label = self._tf_exec(self.next_element)
            self.current_idx += 1
            # example = torch.as_tensor(example)
            # label = torch.as_tensor(example)
        except tf.errors.OutOfRangeError:
            self.reset()
            raise StopIteration

        example = self.transform_sample(example) \
            if self.transform_sample is not None \
            else example
        label = self.transform_label(label) \
            if self.transform_label is not None \
            else label
        return example, label

    def _tf_exec(self, args):
        # Nice try but ingestion doesn't play nice with eager execution
        return args if tf.executing_eagerly() else self.session.run(args)

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        self.current_idx = 0
        return self

    def scan_all(self, max_samples=None):
        # Same as scan but extracts the min/max shape and checks
        # if the dataset is multilabeled
        min_shape = (np.Inf, np.Inf, np.Inf, np.Inf)
        max_shape = (-np.Inf, -np.Inf, -np.Inf, -np.Inf)
        shape_list = []
        is_multilabel = False
        count = 0
        self.reset()
        while count != max_samples:
            try:
                example, label = self._tf_exec(self.next_element)
            except tf.errors.OutOfRangeError:
                self.reset()
                break
            shape_list.append(example.shape)
            min_shape = np.minimum(min_shape, example.shape)
            max_shape = np.maximum(max_shape, example.shape)
            count += 1
            if np.sum(label) > 1:
                is_multilabel = True

        self.num_samples = self.num_samples if max_samples is not None else count
        self.num_classes = label.shape[0]
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.median_shape = np.median(shape_list, axis=0)
        self.mean_shape = np.mean(shape_list, axis=0)
        self.std_shape = np.std(shape_list, axis=0)
        self.is_multilabel = is_multilabel = is_multilabel
        self.reset()


class TFDataLoader(object):
    # I got tired of their implementation...
    __initialized = False

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=default_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    'batch_sampler option is mutually exclusive '
                    'with batch_size, shuffle, sampler, and '
                    'drop_last'
                )

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with ' 'shuffle')

        if self.num_workers < 0:
            raise ValueError(
                'num_workers option cannot be negative; '
                'use num_workers=0 to disable multiprocessing.'
            )

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('sampler'):
            raise ValueError(
                '{} attribute should not be set after {} is '
                'initialized'.format(attr, self.__class__.__name__)
            )
        super(TFDataLoader, self).__setattr__(attr, val)

    def __len__(self):
        return len(self.batch_sampler)

    @property
    def batch_size(self):
        if not hasattr(self, 'batch_sampler'):
            return None
        return self.batch_sampler.batch_size

    @batch_size.setter
    def batch_size(self, val):
        if not hasattr(self, 'batch_sampler'):
            return
        self.batch_sampler = torch.utils.data.sampler.BatchSampler(
            self.sampler, val, self.batch_sampler.drop_last
        )

    @property
    def drop_last(self):
        if not hasattr(self, 'batch_sampler'):
            return None
        return self.batch_sampler.drop_last

    @drop_last.setter
    def drop_last(self, val):
        if not hasattr(self, 'batch_sampler'):
            return
        self.batch_sampler = torch.utils.data.sampler.BatchSampler(
            self.sampler, self.batch_sampler.batch_size, val
        )

    def __iter__(self):
        return _TFDataLoaderIter(self)


class _TFDataLoaderIter(object):
    '''
    This is a copy of the original _DataLoaderIter used by torch
    This version is hacked so it should not need to know how long
    the used dataset is beforehand
    '''
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may alreay be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children. Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed. (Hooks freeing those
    #      resources are registered at importing the Python core libraries at
    #      the top of this file.) So in `__del__`, we check if
    #      `_utils.python_exit_status` is set or `None` (freed), and perform
    #      no-op if so.
    #
    #      Another problem with `__del__` is also related to the library cleanup
    #      calls. When a process ends, it shuts the all its daemonic children
    #      down with a SIGTERM (instead of joining them without a timeout).
    #      Simiarly for threads, but by a different mechanism. This fact,
    #      together with a few implementation details of multiprocessing, forces
    #      us to make workers daemonic. All of our problems arise when a
    #      DataLoader is used in a subprocess, and are caused by multiprocessing
    #      code which looks more or less like this:
    #
    #          try:
    #              your_function_using_a_dataloader()
    #          finally:
    #              multiprocessing.util._exit_function()
    #
    #      The joining/termination mentioned above happens inside
    #      `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #      throws, the stack trace stored in the exception will prevent the
    #      frame which uses `DataLoaderIter` to be freed. If the frame has any
    #      reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #      its  `__del__`, which starts the shutdown procedure, will not be
    #      called. That, in turn, means that workers aren't notified. Attempting
    #      to join in `_exit_function` will then result in a hang.
    #
    #      For context, `_exit_function` is also registered as an `atexit` call.
    #      So it is unclear to me (@ssnl) why this is needed in a finally block.
    #      The code dates back to 2008 and there is no comment on the original
    #      PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #      the finally block and the `atexit` registration) that explains this.
    #
    #      Another choice is to just shutdown workers with logic in 1 above
    #      whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_batch()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           exits.
    #
    #           However, in case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever. Therefore,
    #           for both `worker_result_queue` (worker -> main process/pin_memory_thread)
    #           and each `index_queue` (main process -> worker), we use
    #           `q.cancel_join_thread()` in sender process before any `q.put` to
    #           prevent this automatic join.
    #
    #           Moreover, having all queues called `cancel_join_thread` makes
    #           implementing graceful shutdown logic in `__del__` much easier.
    #           It won't need to get from any queue, which would also need to be
    #           guarded by periodic status checks.
    #
    #           Note that this may leave corrupted data in the queue, but we
    #           don't care about the data anyways once we are shutting down.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # [worker processes]
    #   While loader process is alive:
    #     Get from index_queue.
    #       If got a `None`, exit.
    #       If get anything else,
    #          Check `done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data.
    #       If timed out,
    #          No matter `done_event` is set (still need to see `None`) or not,
    #          must continue to next iteration .
    #
    # [pin_memory_thread]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While True:
    #     Get from index_queue.
    #       If got a `None`, exit.
    #       If get anything else,
    #          Check `done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [main process]
    #   In the DataLoader Iter's `__del__`
    #     a. Set `done_event` (shared with `pin_memory_thread` and workers).
    #
    #        Note: from here on, the workers & `pin_memory_thread` may exit at
    #              any time after they receive `None`.
    #
    #     b. Exit `pin_memory_thread`
    #          i.   Put `None` in `worker_result_queue`.
    #          ii.  Join the `pin_memory_thread`.
    #
    #     c. Exit the workers.
    #          i.   Put `None` in each worker's `index_queue`.
    #          ii.  Join the workers.
    #
    #        NOTE: This has to be after (b) because it may leave corrupted data
    #              in `worker_result_queue`, which `pin_memory_thread` reads
    #              from.
    #
    #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = multiprocessing.Event()

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()
                w = multiprocessing.Process(
                    target=_utils.worker._worker_loop,
                    args=(
                        self.dataset, index_queue, self.worker_result_queue,
                        self.done_event, self.collate_fn, base_seed + i,
                        self.worker_init_fn, i
                    )
                )
                w.daemon = True
                # NB: Process.start() actually take some time as it needs to
                #     start a process and pass the arguments over via a pipe.
                #     Therefore, we only add a worker to self.workers list after
                #     it started, so that we do not call .join() if program dies
                #     before it starts, and __del__ tries to join but will get:
                #     AssertionError: can only join a started process.
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            if self.pin_memory:
                self.data_queue = queue.Queue()
                pin_memory_thread = threading.Thread(
                    target=_utils.pin_memory._pin_memory_loop,
                    args=(
                        self.worker_result_queue, self.data_queue,
                        torch.cuda.current_device(), self.done_event
                    )
                )
                pin_memory_thread.daemon = True
                pin_memory_thread.start()
                # Similar to workers (see comment above), we only register
                # pin_memory_thread once it is started.
                self.pin_memory_thread = pin_memory_thread
            else:
                self.data_queue = self.worker_result_queue

            _utils.signal_handling._set_worker_pids(
                id(self), tuple(w.pid for w in self.workers)
            )
            _utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()
        self._end_reached = False

    def __len__(self):
        return len(self.batch_sampler)

    def _try_get_batch(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `data_queue` for a given timeout. This can
        # also be used as inner loop of fetching without timeout, with the
        # sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self.data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            if not all(w.is_alive() for w in self.workers):
                pids_str = ', '.join(str(w.pid) for w in self.workers if not w.is_alive())
                raise RuntimeError(
                    'DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)
                )
            if isinstance(e, queue.Empty):
                return (False, None)
            raise

    def _get_batch(self):
        # Fetches data from `self.data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_batch(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self.timeout > 0:
            success, data = self._try_get_batch(self.timeout)
            if success:
                return data
            else:
                raise RuntimeError(
                    'DataLoader timed out after {} seconds'.format(self.timeout)
                )
        elif self.pin_memory:
            while self.pin_memory_thread.is_alive():
                success, data = self._try_get_batch()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self.data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_batch()
                if success:
                    return data

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            if self._end_reached:
                self._end_reached = False
                raise StopIteration
            indices = next(self.sample_iter)  # may raise StopIteration
            entries = []
            for i in indices:
                try:
                    entries.append(self.dataset[i])
                except StopIteration:
                    self._end_reached = True
                    if len(entries) == 0:
                        raise StopIteration
                    break
            batch = self.collate_fn(entries)
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, _utils.ExceptionWrapper):
            # make multiline KeyError msg readable by working around
            # a python bug https://bugs.python.org/issue2651
            if batch.exc_type == KeyError and "\n" in batch.exc_msg:
                raise Exception("KeyError:" + batch.exc_msg)
            else:
                raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataLoaderIter cannot be pickled")

    def _shutdown_workers(self):
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self.shutdown:
            self.shutdown = True
            try:
                self.done_event.set()

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, 'pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    # First time do `worker_result_queue.put` in this process.

                    # `cancel_join_thread` in case that `pin_memory_thread` exited.
                    self.worker_result_queue.cancel_join_thread()
                    self.worker_result_queue.put(None)
                    self.pin_memory_thread.join()
                    # Indicate that no more data will be put on this queue by the
                    # current process. This **must** be called after
                    # `pin_memory_thread` is joined because that thread shares the
                    # same pipe handles with this loader thread. If the handle is
                    # closed, Py3 will error in this case, but Py2 will just time
                    # out even if there is data in the queue.
                    self.worker_result_queue.close()

                # Exit workers now.
                for q in self.index_queues:
                    q.put(None)
                    # Indicate that no more data will be put on this queue by the
                    # current process.
                    q.close()
                for w in self.workers:
                    w.join()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self.worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self.worker_pids_set = False

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()
