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
"""Dataset Utils for tfxio."""

from typing import List, Optional, Text

import tensorflow as tf


def detect_compression_type(file_patterns: tf.Tensor) -> tf.Tensor:
  """A TF function that detects compression type given file patterns.

  It simply looks at the names (extensions) of files matching the patterns.

  Args:
    file_patterns: A 1-D string tensor that contains the file patterns.

  Returns:
    A scalar string tensor. The contents are either "GZIP", "" or
    "INVALID_MIXED_COMPRESSION_TYPES".
  """
  # Implementation notes:
  # Because the result of this function usually feeds to another
  # function that creates a TF dataset, and the whole dataset creating logic
  # is usually warpped in an input_fn in the trainer, this function must
  # be a pure composition of TF ops. To be compatible with
  # tf.compat.v1.make_oneshot_dataset, this function cannot be a @tf.function,
  # and it cannot contain any conditional / stateful op either.
  # Once we decide to stop supporting TF 1.x and tf.compat.v1, we can rewrite
  # this as a @tf.function, and use tf.cond / tf.case to make the logic more
  # readable.

  files = tf.io.matching_files(file_patterns)
  is_gz = tf.strings.regex_full_match(files, r".*\.gz$")
  all_files_are_not_gz = tf.math.reduce_all(~is_gz)
  all_files_are_gz = tf.math.reduce_all(is_gz)
  # Encode the 4 cases as integers 0b00 - 0b11 where
  # `all_files_are_not_gz` is bit 0
  # `all_files_are_gz` is bit 1
  # 00: invalid, some files are gz some files are not
  # 01: all are not gz
  # 10: all are gz
  # 11: the only possibility is `files` is empty, can be arbitrary.
  formats = tf.constant(["INVALID_MIXED_COMPRESSION_TYPES", "", "GZIP", ""])
  index = (
      tf.bitwise.left_shift(tf.cast(all_files_are_gz, tf.int32), 1) +
      tf.cast(all_files_are_not_gz, tf.int32))

  return formats[index]


def make_tf_record_dataset(
    file_pattern: List[Text],
    batch_size: int,
    drop_final_batch: bool,
    num_epochs: Optional[int],
    shuffle: Optional[bool],
    shuffle_buffer_size: int,
    shuffle_seed: Optional[int],
    reader_num_threads: int = tf.data.experimental.AUTOTUNE,
    sloppy_ordering: bool = False,
) -> tf.data.Dataset:
  """Returns an interleaved TFRecordDataset with basic options.

  This implementation is a simplified version of
  tf.data.experimental.ops.make_tf_record_dataset().

  Args:
    file_pattern: One or a list of glob patterns. If a list, must not be empty.
    batch_size: An int representing the number of records to combine in a single
      batch.
    drop_final_batch: If `True`, and the batch size does not evenly divide the
      input dataset size, the final smaller batch will be dropped. Defaults to
      `False`.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever. Defaults to `None`.
    shuffle: A boolean, indicates whether the input should be shuffled. Defaults
      to `True`.
    shuffle_buffer_size: Buffer size of the items to shuffle. The size is the
      number of items (i.e. records for a record based TFXIO) to hold. Only data
      read into the buffer will be shuffled (there is no shuffling across
      buffers). A large capacity ensures better shuffling but would increase
      memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    reader_num_threads: Number of threads used to read records. If >1, the
      results will be interleaved. Defaults to tf.data.experimental.AUTOTUNE.
    sloppy_ordering: If `True`, reading performance will be improved at the
      cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to False.
  """
  file_pattern = tf.convert_to_tensor(file_pattern)

  dataset = tf.data.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)

  compression_type = detect_compression_type(file_pattern)
  dataset = dataset.interleave(
      lambda filename: tf.data.TFRecordDataset(filename, compression_type),
      num_parallel_calls=reader_num_threads)
  options = tf.data.Options()
  options.experimental_deterministic = not sloppy_ordering
  dataset = dataset.with_options(options)

  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)

  drop_final_batch = drop_final_batch or num_epochs is None

  return dataset.batch(batch_size, drop_remainder=drop_final_batch)
