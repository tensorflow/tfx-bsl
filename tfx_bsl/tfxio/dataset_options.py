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
"""tfxio.TensorflowDataset options."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
from typing import Optional, Text


class TensorFlowDatasetOptions(
    collections.namedtuple('TensorFlowDatasetOptions', [
        'batch_size', 'drop_final_batch', 'num_epochs', 'shuffle',
        'shuffle_buffer_size', 'shuffle_seed', 'label_key'
    ])):
  """Options for TFXIO's TensorFlowDataset.

  Note: not all of these options may be effective. It depends on the particular
  TFXIO's implementation.
  """

  def __new__(cls,
              batch_size: int,
              drop_final_batch: bool = False,
              num_epochs: Optional[int] = None,
              shuffle: Optional[bool] = None,
              shuffle_buffer_size: Optional[int] = None,
              shuffle_seed: Optional[int] = None,
              label_key: Optional[Text] = None):
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
      shuffle_buffer_size: Buffer size of the ShuffleDataset. A large capacity
        ensures better shuffling but would increase memory usage and startup
        time.
      shuffle_seed: Randomization seed to use for shuffling.
      label_key: name of the label tensor. If provided, the returned dataset
        will yield Tuple[Dict[Text, Tensor], Tensor], where the second term in
        the tuple is the label tensor and the dict (the first term) will not
        contain the label feature.
    """
    return super(TensorFlowDatasetOptions,
                 cls).__new__(cls, batch_size, drop_final_batch, num_epochs,
                              shuffle, shuffle_buffer_size, shuffle_seed,
                              label_key)
