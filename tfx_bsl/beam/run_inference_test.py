# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx_bsl.run_inference."""

import base64
from http import client as http_client
import json
import os
import pickle
from typing import Any, Callable, Optional
from unittest import mock

import apache_beam as beam
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.utils import shared

from googleapiclient import discovery
from googleapiclient import http
import tensorflow as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tfx_bsl.beam import run_inference
from tfx_bsl.beam import test_helpers
from tfx_bsl.public.proto import model_spec_pb2

from google.protobuf import text_format
from absl.testing import parameterized
from tensorflow_serving.apis import prediction_log_pb2


class RunInferenceFixture(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._predict_examples = [
        text_format.Parse(
            """
              features {
                feature { key: "input1" value { float_list { value: 0 }}}
              }
              """, tf.train.Example()),
        text_format.Parse(
            """
              features {
                feature { key: "input1" value { float_list { value: 1 }}}
              }
              """, tf.train.Example()),
    ]

  def _make_beam_pipeline(self):
    return beam.Pipeline(**test_helpers.make_test_beam_pipeline_kwargs())

  def _get_output_data_dir(self, sub_dir=None):
    test_dir = self._testMethodName
    path = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        test_dir)
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    if sub_dir is not None:
      path = os.path.join(path, sub_dir)
    return path

  def _get_results(self, prediction_log_path):
    result = []
    for f in tf.io.gfile.glob(prediction_log_path + '-?????-of-?????'):
      record_iterator = tf.compat.v1.io.tf_record_iterator(path=f)
      for record_string in record_iterator:
        result.append(pickle.loads(record_string))
    return result

  def _prepare_predict_examples(self, example_path):
    with tf.io.TFRecordWriter(example_path) as output_file:
      for example in self._predict_examples:
        output_file.write(example.SerializeToString())

  def _build_predict_model(self, model_path):
    """Exports the dummy sum predict model."""

    with tf.compat.v1.Graph().as_default():
      input_tensors = {
          'x':
              tf.compat.v1.io.FixedLenFeature([1],
                                              dtype=tf.float32,
                                              default_value=0)
      }
      serving_receiver = (
          tf_estimator.export.build_parsing_serving_input_receiver_fn(
              input_tensors)())
      output_tensors = {'y': serving_receiver.features['x'] * 2}
      sess = tf.compat.v1.Session()
      sess.run(tf.compat.v1.initializers.global_variables())
      signature_def = tf_estimator.export.PredictOutput(
          output_tensors).as_signature_def(serving_receiver.receiver_tensors)
      builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(model_path)
      builder.add_meta_graph_and_variables(
          sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
          signature_def_map={
              tf.compat.v1.saved_model.signature_constants
              .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  signature_def,
          })
      builder.save()


_RUN_OFFLINE_INFERENCE_TEST_CASES = [{
    'testcase_name': 'unkeyed_encoded',
    'keyed_input': False,
    'decode_examples': False,
    'use_create_model_handler_beam_api': False
}, {
    'testcase_name': 'keyed_encoded',
    'keyed_input': True,
    'decode_examples': False,
    'use_create_model_handler_beam_api': False
}, {
    'testcase_name': 'unkeyed_decoded',
    'keyed_input': False,
    'decode_examples': True,
    'use_create_model_handler_beam_api': False
}, {
    'testcase_name': 'keyed_decoded',
    'keyed_input': True,
    'decode_examples': True,
    'use_create_model_handler_beam_api': False
}, {
    'testcase_name': 'create_model_handler_beam_api',
    'keyed_input': False,
    'decode_examples': True,
    'use_create_model_handler_beam_api': True
}]


class RunOfflineInferenceTest(RunInferenceFixture, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._multihead_examples = [
        text_format.Parse(
            """
            features {
              feature {key: "x" value { float_list { value: 0.8 }}}
              feature {key: "y" value { float_list { value: 0.2 }}}
            }
            """, tf.train.Example()),
        text_format.Parse(
            """
            features {
              feature {key: "x" value { float_list { value: 0.6 }}}
              feature {key: "y" value { float_list { value: 0.1 }}}
            }
            """, tf.train.Example()),
    ]

  def _prepare_multihead_examples(self, example_path):
    with tf.io.TFRecordWriter(example_path) as output_file:
      for example in self._multihead_examples:
        output_file.write(example.SerializeToString())

  def _build_regression_signature(self, input_tensor, output_tensor):
    """Helper function for building a regression SignatureDef."""
    input_tensor_info = tf.compat.v1.saved_model.utils.build_tensor_info(
        input_tensor)
    signature_inputs = {
        tf.compat.v1.saved_model.signature_constants.REGRESS_INPUTS:
            input_tensor_info
    }
    output_tensor_info = tf.compat.v1.saved_model.utils.build_tensor_info(
        output_tensor)
    signature_outputs = {
        tf.compat.v1.saved_model.signature_constants.REGRESS_OUTPUTS:
            output_tensor_info
    }
    return tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        tf.compat.v1.saved_model.signature_constants.REGRESS_METHOD_NAME)

  def _build_classification_signature(self, input_tensor, scores_tensor):
    """Helper function for building a classification SignatureDef."""
    input_tensor_info = tf.compat.v1.saved_model.utils.build_tensor_info(
        input_tensor)
    signature_inputs = {
        tf.compat.v1.saved_model.signature_constants.CLASSIFY_INPUTS:
            input_tensor_info
    }
    output_tensor_info = tf.compat.v1.saved_model.utils.build_tensor_info(
        scores_tensor)
    signature_outputs = {
        tf.compat.v1.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
            output_tensor_info
    }
    return tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        tf.compat.v1.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

  def _build_multihead_model(self, model_path):
    with tf.compat.v1.Graph().as_default():
      input_example = tf.compat.v1.placeholder(
          tf.string, name='input_examples_tensor')
      config = {
          'x': tf.compat.v1.io.FixedLenFeature(
              [1], dtype=tf.float32, default_value=0),
          'y': tf.compat.v1.io.FixedLenFeature(
              [1], dtype=tf.float32, default_value=0),
      }
      features = tf.compat.v1.parse_example(input_example, config)
      x = features['x']
      y = features['y']
      sum_pred = x + y
      diff_pred = tf.abs(x - y)
      sess = tf.compat.v1.Session()
      sess.run(tf.compat.v1.initializers.global_variables())
      signature_def_map = {
          'regress_diff':
              self._build_regression_signature(input_example, diff_pred),
          'classify_sum':
              self._build_classification_signature(input_example, sum_pred),
          tf.compat.v1.saved_model.signature_constants
          .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              self._build_regression_signature(input_example, sum_pred)
      }
      builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(model_path)
      builder.add_meta_graph_and_variables(
          sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
          signature_def_map=signature_def_map)
      builder.save()

  def _run_inference_with_beam(
      self,
      example_path: str,
      inference_spec_type: model_spec_pb2.InferenceSpecType,
      prediction_log_path: str,
      keyed_input: bool,
      decode_examples: bool,
      use_create_model_handler_beam_api: bool,
      pre_batch_inputs=False,
      load_override_fn: Optional[Callable[[], Any]] = None):
    with self._make_beam_pipeline() as pipeline:
      if keyed_input:
        key = 'TheKey'
        def verify_key(k, v):
          if k != key:
            raise RuntimeError('Wrong Key %s' % k)
          return v
        maybe_pair_with_key = 'PairWithKey' >> beam.Map(lambda x: (key, x))
        maybe_verify_key = 'VerifyKey' >> beam.MapTuple(verify_key)
      else:
        identity = beam.Map(lambda x: x)
        maybe_pair_with_key = 'NoPairWithKey' >> identity
        maybe_verify_key = 'NoVerifyKey' >> identity
      if decode_examples:
        maybe_decode = tf.train.Example.FromString
      else:
        maybe_decode = lambda x: x
      if pre_batch_inputs:
        maybe_batch = beam.combiners.ToList()
      else:
        maybe_batch = beam.Map(lambda x: x)
      if use_create_model_handler_beam_api:
        tf_model_handler = run_inference.create_model_handler(
            inference_spec_type, None, None)
        with self._make_beam_pipeline() as pipeline:
          _ = (
              pipeline
              | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
              | 'RunInference' >>
              beam.ml.inference.base.RunInference(tf_model_handler)
              | 'WritePredictions' >> beam.io.WriteToTFRecord(
                  prediction_log_path, coder=beam.coders.PickleCoder()))
      else:
        with self._make_beam_pipeline() as pipeline:
          _ = (
              pipeline
              | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
              | 'MaybeDecode' >> beam.Map(maybe_decode)
              | 'MaybeBatch' >> maybe_batch
              | maybe_pair_with_key
              | 'RunInference' >> run_inference.RunInferenceImpl(
                  inference_spec_type, load_override_fn)
              | maybe_verify_key
              | 'WritePredictions' >> beam.io.WriteToTFRecord(
                  prediction_log_path, coder=beam.coders.PickleCoder()))

  @parameterized.named_parameters(_RUN_OFFLINE_INFERENCE_TEST_CASES)
  def test_model_path_invalid(self, keyed_input: bool, decode_examples: bool,
                              use_create_model_handler_beam_api: bool):
    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    with self.assertRaisesRegex(IOError, 'SavedModel file does not exist.*'):
      self._run_inference_with_beam(
          example_path,
          model_spec_pb2.InferenceSpecType(
              saved_model_spec=model_spec_pb2.SavedModelSpec(
                  model_path=self._get_output_data_dir())), prediction_log_path,
          keyed_input, decode_examples, use_create_model_handler_beam_api)

  @parameterized.named_parameters(_RUN_OFFLINE_INFERENCE_TEST_CASES)
  def test_estimator_model_predict(self, keyed_input: bool,
                                   decode_examples: bool,
                                   use_create_model_handler_beam_api: bool):
    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_predict_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)), prediction_log_path, keyed_input,
        decode_examples, use_create_model_handler_beam_api)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    inputs = results[0].predict_log.request.inputs['examples']
    self.assertLen(inputs.string_val, 1)
    self.assertEqual(inputs.string_val[0],
                     self._predict_examples[0].SerializeToString())
    outputs = results[0].predict_log.response.outputs['y']
    self.assertEqual(outputs.dtype, tf.float32)
    self.assertLen(outputs.tensor_shape.dim, 2)
    self.assertEqual(outputs.tensor_shape.dim[0].size, 1)
    self.assertEqual(outputs.tensor_shape.dim[1].size, 1)

  @parameterized.named_parameters(_RUN_OFFLINE_INFERENCE_TEST_CASES)
  def test_classify_model(self, keyed_input: bool, decode_examples: bool,
                          use_create_model_handler_beam_api: bool):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path, signature_name=['classify_sum'])),
        prediction_log_path, keyed_input, decode_examples,
        use_create_model_handler_beam_api)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    classify_log = results[0].classify_log
    self.assertLen(classify_log.request.input.example_list.examples, 1)
    self.assertEqual(classify_log.request.input.example_list.examples[0],
                     self._multihead_examples[0])
    self.assertLen(classify_log.response.result.classifications, 1)
    self.assertLen(classify_log.response.result.classifications[0].classes, 1)
    self.assertAlmostEqual(
        classify_log.response.result.classifications[0].classes[0].score, 1.0)

  @parameterized.named_parameters(_RUN_OFFLINE_INFERENCE_TEST_CASES)
  def test_regress_model(self, keyed_input: bool, decode_examples: bool,
                         use_create_model_handler_beam_api: bool):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path, signature_name=['regress_diff'])),
        prediction_log_path, keyed_input, decode_examples,
        use_create_model_handler_beam_api)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    regress_log = results[0].regress_log
    self.assertLen(regress_log.request.input.example_list.examples, 1)
    self.assertEqual(regress_log.request.input.example_list.examples[0],
                     self._multihead_examples[0])
    self.assertLen(regress_log.response.result.regressions, 1)
    self.assertAlmostEqual(regress_log.response.result.regressions[0].value,
                           0.6)

  @parameterized.named_parameters(_RUN_OFFLINE_INFERENCE_TEST_CASES)
  def testRegressModelSideloaded(self, keyed_input: bool, decode_examples: bool,
                                 use_create_model_handler_beam_api: bool):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)

    called_count = 0

    def load_model():
      nonlocal called_count
      called_count += 1
      result = tf.compat.v1.Session(graph=tf.compat.v1.Graph())
      tf.compat.v1.saved_model.loader.load(result, [tf.saved_model.SERVING],
                                           model_path)
      return result

    shared_handle = shared.Shared()
    def _load_override_fn(unused_path, unused_tags):
      return shared_handle.acquire(load_model)
    # Load the model the first time. It should not be loaded again.
    _ = _load_override_fn('', [])

    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path, signature_name=['regress_diff'])),
        prediction_log_path,
        keyed_input,
        decode_examples,
        use_create_model_handler_beam_api,
        load_override_fn=_load_override_fn)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    regress_log = results[0].regress_log
    self.assertLen(regress_log.request.input.example_list.examples, 1)
    self.assertEqual(regress_log.request.input.example_list.examples[0],
                     self._multihead_examples[0])
    self.assertLen(regress_log.response.result.regressions, 1)
    self.assertAlmostEqual(regress_log.response.result.regressions[0].value,
                           0.6)
    # Ensure that the model load only happened once.
    self.assertEqual(called_count, 1)

  @parameterized.named_parameters(_RUN_OFFLINE_INFERENCE_TEST_CASES)
  def test_multi_inference_model(self, keyed_input: bool, decode_examples: bool,
                                 use_create_model_handler_beam_api: bool):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path,
                signature_name=['regress_diff', 'classify_sum'])),
        prediction_log_path, keyed_input, decode_examples,
        use_create_model_handler_beam_api)
    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    multi_inference_log = results[0].multi_inference_log
    self.assertLen(multi_inference_log.request.input.example_list.examples, 1)
    self.assertEqual(multi_inference_log.request.input.example_list.examples[0],
                     self._multihead_examples[0])
    self.assertLen(multi_inference_log.response.results, 2)
    signature_names = []
    for result in multi_inference_log.response.results:
      signature_names.append(result.model_spec.signature_name)
    self.assertIn('regress_diff', signature_names)
    self.assertIn('classify_sum', signature_names)
    result = multi_inference_log.response.results[0]
    self.assertEqual(result.model_spec.signature_name, 'regress_diff')
    self.assertLen(result.regression_result.regressions, 1)
    self.assertAlmostEqual(result.regression_result.regressions[0].value, 0.6)
    result = multi_inference_log.response.results[1]
    self.assertEqual(result.model_spec.signature_name, 'classify_sum')
    self.assertLen(result.classification_result.classifications, 1)
    self.assertLen(result.classification_result.classifications[0].classes, 1)
    self.assertAlmostEqual(
        result.classification_result.classifications[0].classes[0].score, 1.0)

  @parameterized.named_parameters(_RUN_OFFLINE_INFERENCE_TEST_CASES)
  def test_keras_model_predict(self, keyed_input: bool, decode_examples: bool,
                               use_create_model_handler_beam_api: bool):
    inputs = tf.keras.Input(shape=(1,), name='input1')
    output1 = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output1')(
            inputs)
    output2 = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output2')(
            inputs)
    inference_model = tf.keras.models.Model(inputs, [output1, output2])

    class TestKerasModel(tf.keras.Model):

      def __init__(self, inference_model):
        super().__init__(name='test_keras_model')
        self.inference_model = inference_model

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='inputs')
      ])
      def call(self, serialized_example):
        features = {
            'input1':
                tf.compat.v1.io.FixedLenFeature([1],
                                                dtype=tf.float32,
                                                default_value=0)
        }
        input_tensor_dict = tf.io.parse_example(serialized_example, features)
        return inference_model(input_tensor_dict['input1'])

    model = TestKerasModel(inference_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    model_path = self._get_output_data_dir('model')
    tf.saved_model.save(model, model_path, signatures=model.call)

    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)), prediction_log_path, keyed_input,
        decode_examples, use_create_model_handler_beam_api)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)

  @parameterized.named_parameters([{
      'testcase_name': 'decoded_examples',
      'decode_examples': True
  }, {
      'testcase_name': 'encoded_examples',
      'decode_examples': False
  }])
  def test_telemetry(self, decode_examples: bool):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    inference_spec_type = model_spec_pb2.InferenceSpecType(
        saved_model_spec=model_spec_pb2.SavedModelSpec(
            model_path=model_path, signature_name=['classify_sum']))
    pipeline = self._make_beam_pipeline()
    _ = (
        pipeline
        | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
        | 'MaybeDecode' >> beam.Map(
            lambda x: x if decode_examples else tf.train.Example.FromString(x))
        | 'RunInference' >> run_inference.RunInferenceImpl(inference_spec_type))
    run_result = pipeline.run()
    run_result.wait_until_finish()

    num_inferences = run_result.metrics().query(
        MetricsFilter().with_name('num_inferences'))
    self.assertTrue(num_inferences['counters'])
    self.assertEqual(num_inferences['counters'][0].result, 2)
    inference_request_batch_size = run_result.metrics().query(
        MetricsFilter().with_name('inference_request_batch_size'))
    self.assertTrue(inference_request_batch_size['distributions'])
    self.assertEqual(
        inference_request_batch_size['distributions'][0].result.sum, 2)
    inference_request_batch_byte_size = run_result.metrics().query(
        MetricsFilter().with_name('inference_request_batch_byte_size'))
    self.assertTrue(inference_request_batch_byte_size['distributions'])
    self.assertEqual(
        inference_request_batch_byte_size['distributions'][0].result.sum,
        sum(element.ByteSize() for element in self._multihead_examples))
    inference_batch_latency_micro_secs = run_result.metrics().query(
        MetricsFilter().with_name('inference_batch_latency_micro_secs'))
    self.assertTrue(inference_batch_latency_micro_secs['distributions'])
    self.assertGreaterEqual(
        inference_batch_latency_micro_secs['distributions'][0].result.sum, 0)
    load_model_latency_milli_secs = run_result.metrics().query(
        MetricsFilter().with_name('load_model_latency_milli_secs'))
    self.assertTrue(load_model_latency_milli_secs['distributions'])
    self.assertGreaterEqual(
        load_model_latency_milli_secs['distributions'][0].result.sum, 0)

  def test_model_size_bytes(self):
    self.assertEqual(
        1 << 30,
        run_inference.RunInferenceImpl._model_size_bytes(
            '/the/non-existent-or-inaccesible-file'))

    model_path = self._get_output_data_dir('model')
    self._build_predict_model(model_path)
    # The actual model_size is ~2K, but it might fluctuate a bit (eg due to
    # TF version changes).
    model_size = run_inference.RunInferenceImpl._model_size_bytes(model_path)
    self.assertGreater(model_size, 1000)
    self.assertLess(model_size, 5000)

  def test_estimator_model_predict_batched(self):
    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_predict_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)),
        prediction_log_path,
        keyed_input=True,
        decode_examples=True,
        pre_batch_inputs=True,
        use_create_model_handler_beam_api=False)
    results = self._get_results(prediction_log_path)
    self.assertLen(results, 1)
    self.assertLen(results[0], 2)
    for i, ex in enumerate(self._predict_examples):
      inputs = results[0][i].predict_log.request.inputs['examples']
      self.assertLen(inputs.string_val, 1)
      self.assertEqual(inputs.string_val[0], ex.SerializeToString())
    for i in range(2):
      outputs = results[0][i].predict_log.response.outputs['y']
      self.assertEqual(outputs.dtype, tf.float32)
      self.assertLen(outputs.tensor_shape.dim, 2)
      self.assertEqual(outputs.tensor_shape.dim[0].size, 1)
      self.assertEqual(outputs.tensor_shape.dim[1].size, 1)

  @parameterized.named_parameters([
      dict(
          testcase_name='example',
          input_element=tf.train.Example(),
          output_type=prediction_log_pb2.PredictionLog),
      dict(
          testcase_name='bytes',
          input_element=b'',
          output_type=prediction_log_pb2.PredictionLog),
      dict(
          testcase_name='sequence_example',
          input_element=tf.train.SequenceExample(),
          output_type=prediction_log_pb2.PredictionLog),
      dict(
          testcase_name='keyed_example',
          input_element=('key', tf.train.Example()),
          output_type=beam.typehints.Tuple[str,
                                           prediction_log_pb2.PredictionLog]),
      dict(
          testcase_name='keyed_bytes',
          input_element=('key', b''),
          output_type=beam.typehints.Tuple[str,
                                           prediction_log_pb2.PredictionLog]),
      dict(
          testcase_name='keyed_sequence_example',
          input_element=('key', tf.train.SequenceExample()),
          output_type=beam.typehints.Tuple[str,
                                           prediction_log_pb2.PredictionLog]),
      dict(
          testcase_name='keyed_example_list',
          input_element=('key', [tf.train.Example()]),
          output_type=beam.typehints.Tuple[
              str, beam.typehints.List[prediction_log_pb2.PredictionLog]]),
  ])
  def test_infers_element_type(self, input_element, output_type):
    # TODO(zwestrick): Skip building the model, which is not actually used, or
    # stop using parameterized tests if performance becomes an issue.
    model_path = self._get_output_data_dir('model')
    self._build_predict_model(model_path)
    spec = model_spec_pb2.InferenceSpecType(
        saved_model_spec=model_spec_pb2.SavedModelSpec(model_path=model_path))
    inference_transform = run_inference.RunInferenceImpl(spec)
    with self._make_beam_pipeline() as pipeline:
      inference = (
          pipeline | beam.Create([input_element]) | inference_transform)
      self.assertEqual(inference.element_type, output_type)


_RUN_REMOTE_INFERENCE_TEST_CASES = [{
    'testcase_name': 'keyed_input_false',
    'keyed_input': False
}, {
    'keyed_input': True,
    'testcase_name': 'keyed_input_true'
}]


class RunRemoteInferenceTest(RunInferenceFixture, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.example_path = self._get_output_data_dir('example')
    self._prepare_predict_examples(self.example_path)
    # This is from https://ml.googleapis.com/$discovery/rest?version=v1.
    self._discovery_testdata_dir = os.path.join(
        os.path.join(os.path.dirname(__file__), 'testdata'),
        'ml_discovery.json')

  @staticmethod
  def _make_response_body(content, successful):
    if successful:
      response_dict = {'predictions': content}
    else:
      response_dict = {'error': content}
    return json.dumps(response_dict)

  def _set_up_pipeline(self,
                       inference_spec_type: model_spec_pb2.InferenceSpecType,
                       keyed_input: bool):
    if keyed_input:
      key = 'TheKey'
      def verify_key(k, v):
        if k != key:
          raise RuntimeError('Wrong Key %s' % k)
        return v
      maybe_pair_with_key = 'PairWithKey' >> beam.Map(lambda x: (key, x))
      maybe_verify_key = 'VerifyKey' >> beam.MapTuple(verify_key)
    else:
      identity = beam.Map(lambda x: x)
      maybe_pair_with_key = 'NoPairWithKey' >> identity
      maybe_verify_key = 'NoVerifyKey' >> identity
    self.pipeline = self._make_beam_pipeline()
    self.pcoll = (
        self.pipeline
        | 'ReadExamples' >> beam.io.ReadFromTFRecord(
            self.example_path, coder=beam.coders.ProtoCoder(tf.train.Example))
        | maybe_pair_with_key
        | 'RunInference' >> run_inference.RunInferenceImpl(inference_spec_type)
        | maybe_verify_key)

  def _run_inference_with_beam(self):
    self.pipeline_result = self.pipeline.run()
    self.pipeline_result.wait_until_finish()

  @parameterized.named_parameters(_RUN_REMOTE_INFERENCE_TEST_CASES)
  def test_model_predict(self, keyed_input: bool):
    predictions = [{
        'output_1': [0.901],
        'output_2': [0.997]
    }] * len(self._predict_examples)
    builder = http.RequestMockBuilder({
        'ml.projects.predict':
            (None, self._make_response_body(predictions, successful=True))
    })
    resource = discovery.build(
        'ml',
        'v1',
        http=http.HttpMock(self._discovery_testdata_dir,
                           {'status': http_client.OK}),
        requestBuilder=builder)
    with mock.patch('googleapiclient.discovery.' 'build') as response_mock:
      response_mock.side_effect = lambda service, version: resource
      inference_spec_type = model_spec_pb2.InferenceSpecType(
          ai_platform_prediction_model_spec=model_spec_pb2
          .AIPlatformPredictionModelSpec(
              project_id='test-project',
              model_name='test-model',
          ))
      expected = []
      for example in self._predict_examples:
        prediction_log = prediction_log_pb2.PredictionLog()
        predict_log = prediction_log.predict_log
        input_tensor_proto = predict_log.request.inputs['inputs']
        input_tensor_proto.dtype = tf.string.as_datatype_enum
        input_tensor_proto.tensor_shape.dim.add().size = 1
        input_tensor_proto.string_val.append(example.SerializeToString())
        predict_log.response.outputs['output_1'].CopyFrom(
            tf.make_tensor_proto(values=[0.901], dtype=tf.double, shape=(1, 1)))
        predict_log.response.outputs['output_2'].CopyFrom(
            tf.make_tensor_proto(values=[0.997], dtype=tf.double, shape=(1, 1)))
        expected.append(prediction_log)

      self._set_up_pipeline(inference_spec_type, keyed_input)
      assert_that(self.pcoll, equal_to(expected))
      self._run_inference_with_beam()

  @parameterized.named_parameters(_RUN_REMOTE_INFERENCE_TEST_CASES)
  def test_exception_raised_when_response_body_contains_error_entry(
      self, keyed_input: bool):
    error_msg = 'Base64 decode failed.'
    builder = http.RequestMockBuilder({
        'ml.projects.predict':
            (None, self._make_response_body(error_msg, successful=False))
    })
    resource = discovery.build(
        'ml',
        'v1',
        http=http.HttpMock(self._discovery_testdata_dir,
                           {'status': http_client.OK}),
        requestBuilder=builder)
    with mock.patch('googleapiclient.discovery.' 'build') as response_mock:
      response_mock.side_effect = lambda service, version: resource
      inference_spec_type = model_spec_pb2.InferenceSpecType(
          ai_platform_prediction_model_spec=model_spec_pb2
          .AIPlatformPredictionModelSpec(
              project_id='test-project',
              model_name='test-model',
          ))

      try:
        self._set_up_pipeline(inference_spec_type, keyed_input)
        self._run_inference_with_beam()
      except (ValueError, RuntimeError) as exc:
        actual_error_msg = str(exc)
        self.assertIn(error_msg, actual_error_msg)
      else:
        self.fail('Test was expected to throw an exception')

  @parameterized.named_parameters(_RUN_REMOTE_INFERENCE_TEST_CASES)
  def test_exception_raised_when_project_id_is_empty(self, keyed_input: bool):
    inference_spec_type = model_spec_pb2.InferenceSpecType(
        ai_platform_prediction_model_spec=model_spec_pb2
        .AIPlatformPredictionModelSpec(model_name='test-model',))

    with self.assertRaises(ValueError):
      self._set_up_pipeline(inference_spec_type, keyed_input)
      self._run_inference_with_beam()

  def test_can_format_requests(self):
    predictions = [{
        'output_1': [0.901],
        'output_2': [0.997]
    }] * len(self._predict_examples)
    builder = http.RequestMockBuilder({
        'ml.projects.predict':
            (None, self._make_response_body(predictions, successful=True))
    })
    resource = discovery.build(
        'ml',
        'v1',
        http=http.HttpMock(self._discovery_testdata_dir,
                           {'status': http_client.OK}),
        requestBuilder=builder)
    with mock.patch('googleapiclient.discovery.' 'build') as response_mock:
      response_mock.side_effect = lambda service, version: resource
      inference_spec_type = model_spec_pb2.InferenceSpecType(
          ai_platform_prediction_model_spec=model_spec_pb2
          .AIPlatformPredictionModelSpec(
              project_id='test-project',
              model_name='test-model',
          ))

      example = text_format.Parse(
          """
        features {
          feature { key: "x_bytes" value { bytes_list { value: ["ASa8asdf"] }}}
          feature { key: "x" value { bytes_list { value: "JLK7ljk3" }}}
          feature { key: "y" value { int64_list { value: [1, 2] }}}
          feature { key: "z" value { float_list { value: [4.5, 5, 5.5] }}}
        }
        """, tf.train.Example())

      self.pipeline = self._make_beam_pipeline()
      self.pcoll = (
          self.pipeline
          | 'CreateExamples' >> beam.Create([example])
          | 'RunInference'
          >> run_inference.RunInferenceImpl(inference_spec_type))
      self._run_inference_with_beam()

  def test_request_body_with_binary_data(self):
    example = text_format.Parse(
        """
      features {
        feature { key: "x_bytes" value { bytes_list { value: ["ASa8asdf"] }}}
        feature { key: "x" value { bytes_list { value: "JLK7ljk3" }}}
        feature { key: "y" value { int64_list { value: [1, 2] }}}
        feature { key: "z" value { float_list { value: [4.5, 5, 5.5] }}}
      }
      """, tf.train.Example())
    inference_spec_type = model_spec_pb2.InferenceSpecType(
        ai_platform_prediction_model_spec=model_spec_pb2
        .AIPlatformPredictionModelSpec(
            project_id='test_project',
            model_name='test_model',
            version_name='test_version'))
    remote_model_spec = run_inference.create_model_handler(
        inference_spec_type, None, None)
    result = remote_model_spec._make_instances([example],
                                               [example.SerializeToString()])
    self.assertEqual(result, [
        {
            'x_bytes': {
                'b64': 'QVNhOGFzZGY='
            },
            'x': 'JLK7ljk3',
            'y': [1, 2],
            'z': [4.5, 5, 5.5]
        },
    ])

  def test_request_serialized_example(self):
    examples = [
        text_format.Parse(
            """
      features {
        feature { key: "x_bytes" value { bytes_list { value: ["ASa8asdf"] }}}
        feature { key: "x" value { bytes_list { value: "JLK7ljk3" }}}
        feature { key: "y" value { int64_list { value: [1, 2] }}}
      }
      """, tf.train.Example()),
        text_format.Parse(
            """
      context {
        feature { key: "x_bytes" value { bytes_list { value: ["ASa8asdf"] }}}
        feature { key: "x" value { bytes_list { value: "JLK7ljk3" }}}
        feature { key: "y" value { int64_list { value: [1, 2] }}}
      }
      """, tf.train.SequenceExample()),
    ]
    serialized_examples = [e.SerializeToString() for e in examples]
    inference_spec_type = model_spec_pb2.InferenceSpecType(
        ai_platform_prediction_model_spec=model_spec_pb2
        .AIPlatformPredictionModelSpec(
            project_id='test_project',
            model_name='test_model',
            version_name='test_version',
            use_serialization_config=True))
    remote_model_spec = run_inference.create_model_handler(
        inference_spec_type, None, None)
    result = remote_model_spec._make_instances(examples, serialized_examples)
    self.assertEqual(
        result,
        [{'b64': base64.b64encode(se).decode()} for se in serialized_examples])


class RunInferencePerModelTest(RunInferenceFixture, parameterized.TestCase):

  def test_nonkeyed_nonbatched_input(self):
    examples = [
        text_format.Parse(
            """
              features {
                feature { key: "x" value { float_list { value: 0 }}}
              }
              """, tf.train.Example()),
        text_format.Parse(
            """
              features {
                feature { key: "x" value { float_list { value: 1 }}}
              }
              """, tf.train.Example())
    ]
    model_paths = [self._get_output_data_dir(m) for m in ('model1', 'model2')]
    for model_path in model_paths:
      self._build_predict_model(model_path)
    specs = [
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(model_path=p))
        for p in model_paths
    ]
    with self._make_beam_pipeline() as pipeline:
      predictions = (
          pipeline
          | beam.Create(examples)
          | run_inference.RunInferencePerModelImpl(specs)
          | beam.MapTuple(
              lambda _, p2: p2.predict_log.response.outputs['y'].float_val[0]))
      assert_that(predictions, equal_to([0.0, 2.0]))

  def test_keyed_nonbatched_input(self):
    keyed_examples = [
        ('key1', text_format.Parse(
            """
              features {
                feature { key: "x" value { float_list { value: 0 }}}
              }
              """, tf.train.Example())),
        ('key2', text_format.Parse(
            """
              features {
                feature { key: "x" value { float_list { value: 1 }}}
              }
              """, tf.train.Example()))
    ]
    model_paths = [self._get_output_data_dir(m) for m in ('model1', 'model2')]
    for model_path in model_paths:
      self._build_predict_model(model_path)
    specs = [
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(model_path=p))
        for p in model_paths
    ]
    with self._make_beam_pipeline() as pipeline:
      predictions_table = (
          pipeline
          | beam.Create(keyed_examples)
          | 'RunInferencePerModelTable' >>
          run_inference.RunInferencePerModelImpl(specs)
          | beam.MapTuple(lambda k, predict_logs:  # pylint: disable=g-long-lambda
                          (k, [
                              p.predict_log.response.outputs['y'].float_val[0]
                              for p in predict_logs
                          ])))
      assert_that(
          predictions_table,
          equal_to([('key1', [0.0, 0.0]), ('key2', [2.0, 2.0])]),
          label='AssertTable')

  def test_keyed_batched_input(self):
    keyed_batched_examples = [('key_batch_1', [
        text_format.Parse(
            """
                features {
                  feature { key: "x" value { float_list { value: 0 }}}
                }
              """, tf.train.Example()),
        text_format.Parse(
            """
                features {
                  feature { key: "x" value { float_list { value: 1 }}}
                }
              """, tf.train.Example())
    ]),
                              ('key_batch_2', [
                                  text_format.Parse(
                                      """
                features {
                  feature { key: "x" value { float_list { value: 2 }}}
                }
              """, tf.train.Example()),
                                  text_format.Parse(
                                      """
                features {
                  feature { key: "x" value { float_list { value: 3 }}}
                }
              """, tf.train.Example())
                              ])]
    model_paths = [self._get_output_data_dir(m) for m in ('model1', 'model2')]
    for model_path in model_paths:
      self._build_predict_model(model_path)
    specs = [
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(model_path=p))
        for p in model_paths
    ]
    with self._make_beam_pipeline() as pipeline:
      predictions_table = (
          pipeline
          | beam.Create(keyed_batched_examples)
          | 'RunInferencePerModelTable' >>
          run_inference.RunInferencePerModelImpl(specs)
          | beam.MapTuple(lambda k, batched_predicit_logs:  # pylint: disable=g-long-lambda
                          (k, [[  # pylint: disable=g-complex-comprehension
                              pl.predict_log.response.outputs['y'].float_val[
                                  0] for pl in pls
                          ] for pls in batched_predicit_logs])))
      assert_that(
          predictions_table,
          equal_to([('key_batch_1', [[0.0, 2.0], [0.0, 2.0]]),
                    ('key_batch_2', [[4.0, 6.0], [4.0, 6.0]])]),
          label='AssertTable')

  @parameterized.named_parameters([
      dict(
          testcase_name='example',
          input_element=tf.train.Example(),
          output_type=beam.typehints.Tuple[prediction_log_pb2.PredictionLog,
                                           prediction_log_pb2.PredictionLog]),
      dict(
          testcase_name='bytes',
          input_element=b'',
          output_type=beam.typehints.Tuple[prediction_log_pb2.PredictionLog,
                                           prediction_log_pb2.PredictionLog]),
      dict(
          testcase_name='sequence_example',
          input_element=tf.train.SequenceExample(),
          output_type=beam.typehints.Tuple[prediction_log_pb2.PredictionLog,
                                           prediction_log_pb2.PredictionLog]),
      dict(
          testcase_name='keyed_example',
          input_element=('key', tf.train.Example()),
          output_type=beam.typehints.Tuple[
              str, beam.typehints.Tuple[prediction_log_pb2.PredictionLog,
                                        prediction_log_pb2.PredictionLog]]),
      dict(
          testcase_name='keyed_bytes',
          input_element=('key', b''),
          output_type=beam.typehints.Tuple[
              str, beam.typehints.Tuple[prediction_log_pb2.PredictionLog,
                                        prediction_log_pb2.PredictionLog]]),
      dict(
          testcase_name='keyed_sequence_example',
          input_element=('key', tf.train.SequenceExample()),
          output_type=beam.typehints.Tuple[
              str, beam.typehints.Tuple[prediction_log_pb2.PredictionLog,
                                        prediction_log_pb2.PredictionLog]]),
      dict(
          testcase_name='keyed_batched_examples',
          input_element=('key', [tf.train.Example()]),
          output_type=beam.typehints.Tuple[str, beam.typehints.Tuple[
              beam.typehints.List[prediction_log_pb2.PredictionLog],
              beam.typehints.List[prediction_log_pb2.PredictionLog]]]),
  ])
  def test_infers_element_type(self, input_element, output_type):
    # TODO(zwestrick): Skip building the model, which is not actually used, or
    # stop using parameterized tests if performance becomes an issue.
    model_paths = [self._get_output_data_dir(m) for m in ('model1', 'model2')]
    for model_path in model_paths:
      self._build_predict_model(model_path)
    specs = [
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(model_path=p))
        for p in model_paths
    ]
    inference_transform = run_inference.RunInferencePerModelImpl(specs)
    with self._make_beam_pipeline() as pipeline:
      inference = (
          pipeline | beam.Create([input_element]) | inference_transform)
      self.assertEqual(inference.element_type, output_type)

if __name__ == '__main__':
  tf.test.main()
