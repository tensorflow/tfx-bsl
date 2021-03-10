# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset options for tfxio."""

from typing import NamedTuple, Optional
import tensorflow as tf


class TensorFlowDatasetOptions(
    NamedTuple('TensorFlowDatasetOptions', [
        ('batch_size', int),
        ('drop_final_batch', bool),
        ('num_epochs', Optional[int]),
        ('shuffle', bool),
        ('shuffle_buffer_size', int),
        ('shuffle_seed', Optional[int]),
        ('prefetch_buffer_size', int),
        ('reader_num_threads', int),
        ('parser_num_threads', int),
        ('sloppy_ordering', bool),
        ('label_key', Optional[str]),
    ])):
  """Options for TFXIO's TensorFlowDataset.

  Note: not all of these options may be effective. It depends on the particular
  TFXIO's implementation.
  """

  def __new__(cls,
              batch_size: int,
              drop_final_batch: bool = False,
              num_epochs: Optional[int] = None,
              shuffle: bool = True,
              shuffle_buffer_size: int = 10000,
              shuffle_seed: Optional[int] = None,
              prefetch_buffer_size: int = tf.data.experimental.AUTOTUNE,
              reader_num_threads: int = tf.data.experimental.AUTOTUNE,
              parser_num_threads: int = tf.data.experimental.AUTOTUNE,
              sloppy_ordering: bool = False,
              label_key: Optional[str] = None):
    """Returns a dataset options object.

    Args:
      batch_size: An int representing the number of records to combine in a
        single batch.
      drop_final_batch: If `True`, and the batch size does not evenly divide the
        input dataset size, the final smaller batch will be dropped. Defaults to
        `False`.
      num_epochs: Integer specifying the number of times to read through the
        dataset. If None, cycles through the dataset forever. Defaults to
        `None`.
      shuffle: A boolean, indicates whether the input should be shuffled.
        Defaults to `True`.
      shuffle_buffer_size: Buffer size of the items to shuffle. The size is the
        number of items (i.e. records for a record based TFXIO) to hold. Only
        data read into the buffer will be shuffled (there is no shuffling across
        buffers). A large capacity ensures better shuffling but would increase
        memory usage and startup time.
      shuffle_seed: Randomization seed to use for shuffling.
      prefetch_buffer_size: Number of feature batches to prefetch in order to
        improve performance. Recommended value is the number of batches consumed
        per training step. Defaults to auto-tune.
      reader_num_threads: Number of threads used to read records. If >1, the
        results will be interleaved. Defaults to tf.data.experimental.AUTOTUNE.
      parser_num_threads: Number of threads to use for parsing `Example` tensors
        into a dictionary of `Feature` tensors (if applicable). Defaults to
        auto-tune.
      sloppy_ordering: If `True`, reading performance will be improved at the
        cost of non-deterministic ordering. If `False`, the order of elements
        produced is deterministic prior to shuffling (elements are still
        randomized if `shuffle=True`. Note that if the seed is set, then order
        of elements after shuffling is deterministic). Defaults to False.
      label_key: name of the label tensor. If provided, the returned dataset
        will yield Tuple[Dict[str, Tensor], Tensor], where the second term in
        the tuple is the label tensor and the dict (the first term) will not
        contain the label feature.
    """
    return super().__new__(cls, batch_size, drop_final_batch, num_epochs,
                           shuffle, shuffle_buffer_size, shuffle_seed,
                           prefetch_buffer_size, reader_num_threads,
                           parser_num_threads, sloppy_ordering, label_key)


class RecordBatchesOptions(
    NamedTuple('RecordBatchesOptions', [('batch_size', int),
                                        ('drop_final_batch', bool),
                                        ('num_epochs', Optional[int]),
                                        ('shuffle', bool),
                                        ('shuffle_buffer_size', int),
                                        ('shuffle_seed', Optional[int])])):
  """Options for TFXIO's RecordBatches.

  Note: not all of these options may be effective. It depends on the particular
  TFXIO's implementation.
  """

  def __new__(cls,
              batch_size: int,
              drop_final_batch: bool = False,
              num_epochs: Optional[int] = None,
              shuffle: bool = True,
              shuffle_buffer_size: int = 10000,
              shuffle_seed: Optional[int] = None):
    """Returns a dataset options object.

    Args:
      batch_size: An int representing the number of records to combine in a
        single batch.
      drop_final_batch: If `True`, and the batch size does not evenly divide the
        input dataset size, the final smaller batch will be dropped. Defaults to
        `False`.
      num_epochs: Integer specifying the number of times to read through the
        dataset. If None, cycles through the dataset forever. Defaults to
        `None`.
      shuffle: A boolean, indicates whether the input should be shuffled.
        Defaults to `True`.
      shuffle_buffer_size: Buffer size of the items to shuffle. The size is the
        number of items (i.e. records for a record based TFXIO) to hold. Only
        data read into the buffer will be shuffled (there is no shuffling across
        buffers). A large capacity ensures better shuffling but would increase
        memory usage and startup time.
      shuffle_seed: Randomization seed to use for shuffling.
    """
    return super(RecordBatchesOptions,
                 cls).__new__(cls, batch_size, drop_final_batch, num_epochs,
                              shuffle, shuffle_buffer_size, shuffle_seed)
