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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import base64
import json
import os
try:
  import unittest.mock as mock
except ImportError:
  import mock

import apache_beam as beam
import pyarrow as pa
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from googleapiclient import discovery
from googleapiclient import http
from six.moves import http_client
import tensorflow as tf
from tfx_bsl.beam import bsl_util
from tfx_bsl.beam import run_inference
from tfx_bsl.beam.bsl_constants import DataType
from tfx_bsl.beam.bsl_constants import _RECORDBATCH_COLUMN
from tfx_bsl.public.proto import model_spec_pb2
from tfx_bsl.tfxio import test_util
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record

from google.protobuf import text_format
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_metadata.proto.v0 import schema_pb2


class RunInferenceFixture(tf.test.TestCase):

  def setUp(self):
    super(RunInferenceFixture, self).setUp()
    self._predict_examples = [
        text_format.Parse(
            """
            features {
              feature { key: "input1" value { float_list { value: 0 }}}
            }
            """, tf.train.Example()),
    ]

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

  def _prepare_predict_examples(self, example_path):
    with tf.io.TFRecordWriter(example_path) as output_file:
      for example in self._predict_examples:
        output_file.write(example.SerializeToString())


class RunOfflineInferenceExamplesTest(RunInferenceFixture):

  def setUp(self):
    super(RunOfflineInferenceExamplesTest, self).setUp()
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

    self.schema = text_format.Parse(
      """
      tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "x"
            value {
              dense_tensor {
                column_name: "x"
                shape { dim { size: 1 } }
              }
            }
          }
          tensor_representation {
            key: "y"
            value {
              dense_tensor {
                column_name: "y"
                shape { dim { size: 1 } }
              }
            }
          }
        }
      }
      feature {
        name: "x"
        type: FLOAT
      }
      feature {
        name: "y"
        type: FLOAT
      }
      """, schema_pb2.Schema())

  def _prepare_multihead_examples(self, example_path):
    with tf.io.TFRecordWriter(example_path) as output_file:
      for example in self._multihead_examples:
        output_file.write(example.SerializeToString())

  def _run_inference_with_beam(self, example_path, inference_spec_type,
                               prediction_log_path, include_schema = False):
    schema = None
    if include_schema:
      schema = self.schema

    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
          | 'ParseExamples' >> beam.Map(tf.train.Example.FromString)
          | 'RunInference' >> run_inference.RunInferenceOnExamples(
              inference_spec_type, schema=schema)
          | 'WritePredictions' >> beam.io.WriteToTFRecord(
              prediction_log_path,
              coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))

  def _get_results(self, prediction_log_path):
    results = []
    for f in tf.io.gfile.glob(prediction_log_path + '-?????-of-?????'):
      record_iterator = tf.compat.v1.io.tf_record_iterator(path=f)
      for record_string in record_iterator:
        prediction_log = prediction_log_pb2.PredictionLog()
        prediction_log.MergeFromString(record_string)
        results.append(prediction_log)
    return results


  def testKerasModelPredict(self):
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
        super(TestKerasModel, self).__init__(name='test_keras_model')
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
    tf.compat.v1.keras.experimental.export_saved_model(
        model, model_path, serving_only=True)

    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)), prediction_log_path)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)

  def testKerasModelPredictMultiTensor(self):
    input1 = tf.keras.layers.Input((1,), name='x')
    input2 = tf.keras.layers.Input((1,), name='y')

    x1 = tf.keras.layers.Dense(10)(input1)
    x2 = tf.keras.layers.Dense(10)(input2)
    output = tf.keras.layers.Dense(5, name='output')(x2)

    model = tf.keras.models.Model([input1, input2], output)
    model_path = self._get_output_data_dir('model')
    tf.compat.v1.keras.experimental.export_saved_model(
        model, model_path, serving_only=True)

    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)),
        prediction_log_path, include_schema = True)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    for result in results:
      self.assertLen(result.predict_log.request.inputs, 2)
      self.assertAllInSet(list(result.predict_log.request.inputs), list(['x','y']))


class RunRemoteInferenceExamplesTest(RunInferenceFixture):

  def setUp(self):
    super(RunRemoteInferenceExamplesTest, self).setUp()
    self._example_path = self._get_output_data_dir('example')
    self._prepare_predict_examples(self._example_path)
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

  def _set_up_pipeline(self, inference_spec_type):
    self.pipeline = beam.Pipeline()
    self.pcoll = (
        self.pipeline
        | 'ReadExamples' >> beam.io.ReadFromTFRecord(self._example_path)
        | 'ParseExamples' >> beam.Map(tf.train.Example.FromString)
        | 'RunInference' >> run_inference.RunInferenceOnExamples(inference_spec_type))

  def _run_inference_with_beam(self):
    self.pipeline_result = self.pipeline.run()
    self.pipeline_result.wait_until_finish()

  def test_model_predict(self):
    predictions = [{'output_1': [0.901], 'output_2': [0.997]}]
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

      prediction_log = prediction_log_pb2.PredictionLog()
      prediction_log.predict_log.response.outputs['output_1'].CopyFrom(
          tf.make_tensor_proto(values=[0.901], dtype=tf.double, shape=(1, 1)))
      prediction_log.predict_log.response.outputs['output_2'].CopyFrom(
          tf.make_tensor_proto(values=[0.997], dtype=tf.double, shape=(1, 1)))

      self._set_up_pipeline(inference_spec_type)
      assert_that(self.pcoll, equal_to([prediction_log]))
      self._run_inference_with_beam()


class RunOfflineInferenceSequenceExamplesTest(RunInferenceFixture):

  def setUp(self):
    super(RunOfflineInferenceSequenceExamplesTest, self).setUp()
    self._predict_examples = [
        text_format.Parse(
            """
            context {
              feature { key: "input1" value { float_list { value: 0 }}}
            }
            """, tf.train.SequenceExample()),
        text_format.Parse(
            """
            context {
              feature { key: "input1" value { float_list { value: 1 }}}
            }
            """, tf.train.SequenceExample()),
    ]
    self._multihead_examples = [
        text_format.Parse(
            """
            context {
              feature {key: "x" value { float_list { value: 0.8 }}}
              feature {key: "y" value { float_list { value: 0.2 }}}
            }
            """, tf.train.SequenceExample()),
        text_format.Parse(
            """
            context {
              feature {key: "x" value { float_list { value: 0.6 }}}
              feature {key: "y" value { float_list { value: 0.1 }}}
            }
            """, tf.train.SequenceExample()),
    ]

    self.schema = text_format.Parse(
      """
      tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "x"
            value {
              dense_tensor {
                column_name: "x"
                shape { dim { size: 1 } }
              }
            }
          }
          tensor_representation {
            key: "y"
            value {
              dense_tensor {
                column_name: "y"
                shape { dim { size: 1 } }
              }
            }
          }
        }
      }
      feature {
        name: "x"
        type: FLOAT
      }
      feature {
        name: "y"
        type: FLOAT
      }
      """, schema_pb2.Schema())

  def _prepare_multihead_examples(self, example_path):
    with tf.io.TFRecordWriter(example_path) as output_file:
      for example in self._multihead_examples:
        output_file.write(example.SerializeToString())

  def _run_inference_with_beam(self, example_path, inference_spec_type,
                               prediction_log_path, include_schema = False):
    schema = None
    if include_schema:
      schema = self.schema

    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
          | 'ParseExamples' >> beam.Map(tf.train.SequenceExample.FromString)
          | 'RunInference' >> run_inference.RunInferenceOnExamples(
              inference_spec_type, schema=schema)
          | 'WritePredictions' >> beam.io.WriteToTFRecord(
              prediction_log_path,
              coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))

  def _get_results(self, prediction_log_path):
    results = []
    for f in tf.io.gfile.glob(prediction_log_path + '-?????-of-?????'):
      record_iterator = tf.compat.v1.io.tf_record_iterator(path=f)
      for record_string in record_iterator:
        prediction_log = prediction_log_pb2.PredictionLog()
        prediction_log.MergeFromString(record_string)
        results.append(prediction_log)
    return results


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


  def testClassifyModelError(self):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    error_msg = 'Operation type'
    try:
      self._run_inference_with_beam(
          example_path,
          model_spec_pb2.InferenceSpecType(
              saved_model_spec=model_spec_pb2.SavedModelSpec(
                  model_path=model_path, signature_name=['classify_sum'])),
          prediction_log_path)
    except ValueError as exc:
        actual_error_msg = str(exc)
        self.assertTrue(actual_error_msg.startswith(error_msg))
    else:
      self.fail('Test was expected to throw ValueError exception')

  def testRegressModelError(self):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    error_msg = 'Operation type'
    try:
      self._run_inference_with_beam(
          example_path,
          model_spec_pb2.InferenceSpecType(
              saved_model_spec=model_spec_pb2.SavedModelSpec(
                  model_path=model_path, signature_name=['regress_diff'])),
          prediction_log_path)
    except ValueError as exc:
        actual_error_msg = str(exc)
        self.assertTrue(actual_error_msg.startswith(error_msg))
    else:
      self.fail('Test was expected to throw ValueError exception')

  def testMultiInferenceModelError(self):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    error_msg = 'Operation type'
    try:
      self._run_inference_with_beam(
          example_path,
          model_spec_pb2.InferenceSpecType(
              saved_model_spec=model_spec_pb2.SavedModelSpec(
                  model_path=model_path,
                  signature_name=['regress_diff', 'classify_sum'])),
          prediction_log_path)
    except ValueError as exc:
        actual_error_msg = str(exc)
        self.assertTrue(actual_error_msg.startswith(error_msg))
    else:
      self.fail('Test was expected to throw ValueError exception')


  def testKerasModelPredict(self):
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
        super(TestKerasModel, self).__init__(name='test_keras_model')
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
    tf.compat.v1.keras.experimental.export_saved_model(
        model, model_path, serving_only=True)

    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)), prediction_log_path)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)

  def testKerasModelPredictMultiTensor(self):
    input1 = tf.keras.layers.Input((1,), name='x')
    input2 = tf.keras.layers.Input((1,), name='y')

    x1 = tf.keras.layers.Dense(10)(input1)
    x2 = tf.keras.layers.Dense(10)(input2)
    output = tf.keras.layers.Dense(5, name='output')(x2)

    model = tf.keras.models.Model([input1, input2], output)
    model_path = self._get_output_data_dir('model')
    tf.compat.v1.keras.experimental.export_saved_model(
        model, model_path, serving_only=True)

    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)),
        prediction_log_path, include_schema = True)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    for result in results:
      self.assertLen(result.predict_log.request.inputs, 2)
      self.assertAllInSet(list(result.predict_log.request.inputs), list(['x','y']))


class RunOfflineInferenceArrowTest(RunInferenceFixture):

  def setUp(self):
    super(RunOfflineInferenceArrowTest, self).setUp()
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

    self.schema = text_format.Parse(
      """
      tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "x"
            value {
              dense_tensor {
                column_name: "x"
                shape { dim { size: 1 } }
              }
            }
          }
          tensor_representation {
            key: "y"
            value {
              dense_tensor {
                column_name: "y"
                shape { dim { size: 1 } }
              }
            }
          }
        }
      }
      feature {
        name: "x"
        type: FLOAT
      }
      feature {
        name: "y"
        type: FLOAT
      }
      """, schema_pb2.Schema())

  def _prepare_multihead_examples(self, example_path):
    with tf.io.TFRecordWriter(example_path) as output_file:
      for example in self._multihead_examples:
        output_file.write(example.SerializeToString())

  def _build_predict_model(self, model_path):
    """Exports the dummy sum predict model."""

    with tf.compat.v1.Graph().as_default():
      input_tensors = {
          'x': tf.compat.v1.io.FixedLenFeature(
              [1], dtype=tf.float32, default_value=0)
      }
      serving_receiver = (
          tf.compat.v1.estimator.export.build_parsing_serving_input_receiver_fn(
              input_tensors)())
      output_tensors = {'y': serving_receiver.features['x'] * 2}
      sess = tf.compat.v1.Session()
      sess.run(tf.compat.v1.initializers.global_variables())
      signature_def = tf.compat.v1.estimator.export.PredictOutput(
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

  def _run_inference_with_beam(self, example_path, inference_spec_type,
                               prediction_log_path, include_config = False):
    # test _RunInferenceOnRecordBatch
    converter = tf_example_record.TFExampleBeamRecord(
      physical_format="inmem", telemetry_descriptors=[],
      schema=self.schema, raw_record_column_name=_RECORDBATCH_COLUMN)

    if include_config:
      tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=converter.ArrowSchema(),
        tensor_representations=converter.TensorRepresentations())

      with beam.Pipeline() as pipeline:
        _ = (
          pipeline
          | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
          | 'ConvertToRecordBatch' >> converter.BeamSource()
          | 'RunInference' >> run_inference._RunInferenceOnRecordBatch(
              inference_spec_type, DataType.EXAMPLE, tensor_adapter_config)
          | 'WritePredictions' >> beam.io.WriteToTFRecord(
              prediction_log_path,
              coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))
    else:
      with beam.Pipeline() as pipeline:
        _ = (
          pipeline
          | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
          | 'ConvertToRecordBatch' >> converter.BeamSource()
          | 'RunInference' >> run_inference._RunInferenceOnRecordBatch(
                inference_spec_type, DataType.EXAMPLE)
          | 'WritePredictions' >> beam.io.WriteToTFRecord(
              prediction_log_path,
              coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))

  def _get_results(self, prediction_log_path):
    results = []
    for f in tf.io.gfile.glob(prediction_log_path + '-?????-of-?????'):
      record_iterator = tf.compat.v1.io.tf_record_iterator(path=f)
      for record_string in record_iterator:
        prediction_log = prediction_log_pb2.PredictionLog()
        prediction_log.MergeFromString(record_string)
        results.append(prediction_log)
    return results

  def testModelPathInvalid(self):
    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    with self.assertRaisesRegexp(IOError, 'SavedModel file does not exist.*'):
      self._run_inference_with_beam(
          example_path,
          model_spec_pb2.InferenceSpecType(
              saved_model_spec=model_spec_pb2.SavedModelSpec(
                  model_path=self._get_output_data_dir())), prediction_log_path)

  def testEstimatorModelPredict(self):
    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_predict_model(model_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)), prediction_log_path)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    self.assertEqual(
        results[0].predict_log.request.inputs[
            run_inference._DEFAULT_INPUT_KEY].string_val[0],
        self._predict_examples[0].SerializeToString())
    self.assertEqual(results[0].predict_log.response.outputs['y'].dtype,
                     tf.float32)
    self.assertLen(
        results[0].predict_log.response.outputs['y'].tensor_shape.dim, 2)
    self.assertEqual(
        results[0].predict_log.response.outputs['y'].tensor_shape.dim[0].size,
        1)
    self.assertEqual(
        results[0].predict_log.response.outputs['y'].tensor_shape.dim[1].size,
        1)

  def testClassifyModel(self):
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
        prediction_log_path)

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

  def testRegressModel(self):
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
        prediction_log_path)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    regress_log = results[0].regress_log
    self.assertLen(regress_log.request.input.example_list.examples, 1)
    self.assertEqual(regress_log.request.input.example_list.examples[0],
                     self._multihead_examples[0])
    self.assertLen(regress_log.response.result.regressions, 1)
    self.assertAlmostEqual(regress_log.response.result.regressions[0].value,
                           0.6)

  def testMultiInferenceModel(self):
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
        prediction_log_path)
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

  def testKerasModelPredict(self):
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
        super(TestKerasModel, self).__init__(name='test_keras_model')
        self.inference_model = inference_model

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='inputs')
      ])
      def call(self, serialized_example):
        features = {
            'input1': tf.compat.v1.io.FixedLenFeature(
              [1], dtype=tf.float32,
              default_value=0)
        }
        input_tensor_dict = tf.io.parse_example(serialized_example, features)
        return inference_model(input_tensor_dict['input1'])

    model = TestKerasModel(inference_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    example_path = self._get_output_data_dir('examples')
    self._prepare_predict_examples(example_path)
    model_path = self._get_output_data_dir('model')
    tf.compat.v1.keras.experimental.export_saved_model(
        model, model_path, serving_only=True)

    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)), prediction_log_path)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)

  def testKerasModelPredictMultiTensor(self):
    input1 = tf.keras.layers.Input((1,), name='x')
    input2 = tf.keras.layers.Input((1,), name='y')

    x1 = tf.keras.layers.Dense(10)(input1)
    x2 = tf.keras.layers.Dense(10)(input2)
    output = tf.keras.layers.Dense(5, name='output')(x2)

    model = tf.keras.models.Model([input1, input2], output)
    model_path = self._get_output_data_dir('model')
    tf.compat.v1.keras.experimental.export_saved_model(
        model, model_path, serving_only=True)

    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')
    self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)),
        prediction_log_path, include_config = True)

    results = self._get_results(prediction_log_path)
    self.assertLen(results, 2)
    for result in results:
      self.assertLen(result.predict_log.request.inputs, 2)
      self.assertAllInSet(list(result.predict_log.request.inputs), list(['x','y']))

  def testMultiTensorError(self):
    input1 = tf.keras.layers.Input((1,), name='x')
    input2 = tf.keras.layers.Input((1,), name='y')

    x1 = tf.keras.layers.Dense(10)(input1)
    x2 = tf.keras.layers.Dense(10)(input2)
    output = tf.keras.layers.Dense(5, name='output')(x2)

    model = tf.keras.models.Model([input1, input2], output)
    model_path = self._get_output_data_dir('model')
    tf.compat.v1.keras.experimental.export_saved_model(
        model, model_path, serving_only=True)

    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    prediction_log_path = self._get_output_data_dir('predictions')

    error_msg = 'Tensor adaptor config is required with a multi-input model'
    try:
      self._run_inference_with_beam(
        example_path,
        model_spec_pb2.InferenceSpecType(
            saved_model_spec=model_spec_pb2.SavedModelSpec(
                model_path=model_path)),
              prediction_log_path, include_config = False)
    except ValueError as exc:
      actual_error_msg = str(exc)
      self.assertTrue(actual_error_msg.startswith(error_msg))
    else:
      self.fail('Test was expected to throw ValueError exception')

  def testTelemetry(self):
    example_path = self._get_output_data_dir('examples')
    self._prepare_multihead_examples(example_path)
    model_path = self._get_output_data_dir('model')
    self._build_multihead_model(model_path)
    inference_spec_type = model_spec_pb2.InferenceSpecType(
        saved_model_spec=model_spec_pb2.SavedModelSpec(
            model_path=model_path, signature_name=['classify_sum']))
  
    pipeline = beam.Pipeline()
    converter = tf_example_record.TFExampleBeamRecord(
      physical_format="inmem",
      telemetry_descriptors=[],
      raw_record_column_name=_RECORDBATCH_COLUMN)
    _ = (
        pipeline 
        | 'ReadExamples' >> beam.io.ReadFromTFRecord(example_path)
        | 'ConvertToRecordBatch' >> converter.BeamSource()
        | 'RunInference' >> run_inference._RunInferenceOnRecordBatch(
              inference_spec_type, DataType.EXAMPLE))
    run_result = pipeline.run()
    run_result.wait_until_finish()

    num_inferences = run_result.metrics().query(
        MetricsFilter().with_name('num_inferences'))
    self.assertTrue(num_inferences['counters'])
    self.assertEqual(num_inferences['counters'][0].result, 2)
    num_instances = run_result.metrics().query(
        MetricsFilter().with_name('num_instances'))
    self.assertTrue(num_instances['counters'])
    self.assertEqual(num_instances['counters'][0].result, 2)
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


class RunRemoteInferenceArrowTest(RunInferenceFixture):

  def setUp(self):
    super(RunRemoteInferenceArrowTest, self).setUp()
    self._example_path = self._get_output_data_dir('example')
    self._prepare_predict_examples(self._example_path)
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

  def _set_up_pipeline(self, inference_spec_type):
    self.pipeline = beam.Pipeline()
    converter = tf_example_record.TFExampleBeamRecord(
      physical_format="inmem",
      telemetry_descriptors=[],
      raw_record_column_name=_RECORDBATCH_COLUMN)
    self.pcoll = (
        self.pipeline
        | 'ReadExamples' >> beam.io.ReadFromTFRecord(self._example_path)
        | 'ConvertToRecordBatch' >> converter.BeamSource()
        | 'RunInference' >> run_inference._RunInferenceOnRecordBatch(
              inference_spec_type, DataType.EXAMPLE))

  def _run_inference_with_beam(self):
    self.pipeline_result = self.pipeline.run()
    self.pipeline_result.wait_until_finish()

  def test_model_predict(self):
    predictions = [{'output_1': [0.901], 'output_2': [0.997]}]
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

      prediction_log = prediction_log_pb2.PredictionLog()
      prediction_log.predict_log.response.outputs['output_1'].CopyFrom(
          tf.make_tensor_proto(values=[0.901], dtype=tf.double, shape=(1, 1)))
      prediction_log.predict_log.response.outputs['output_2'].CopyFrom(
          tf.make_tensor_proto(values=[0.997], dtype=tf.double, shape=(1, 1)))

      self._set_up_pipeline(inference_spec_type)
      assert_that(self.pcoll, equal_to([prediction_log]))
      self._run_inference_with_beam()

  def test_exception_raised_when_response_body_contains_error_entry(self):
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
        self._set_up_pipeline(inference_spec_type)
        self._run_inference_with_beam()
      except ValueError as exc:
        actual_error_msg = str(exc)
        self.assertTrue(actual_error_msg.startswith(error_msg))
      else:
        self.fail('Test was expected to throw ValueError exception')

  def test_exception_raised_when_project_id_is_empty(self):
    inference_spec_type = model_spec_pb2.InferenceSpecType(
        ai_platform_prediction_model_spec=model_spec_pb2
        .AIPlatformPredictionModelSpec(model_name='test-model',))

    with self.assertRaises(ValueError):
      self._set_up_pipeline(inference_spec_type)
      self._run_inference_with_beam()

  def test_can_format_requests(self):
    # Ensure _RemotePredictDoFn._prepare_instances produces
    # JSON-serializable objects
    builder = http.RequestMockBuilder({
        'ml.projects.predict': (None,
                                self._make_response_body([], successful=True))
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
          ai_platform_prediction_model_spec=model_spec_pb2.AIPlatformPredictionModelSpec(
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

      converter = tf_example_record.TFExampleBeamRecord(
        physical_format="inmem",
        telemetry_descriptors=[],
        raw_record_column_name=_RECORDBATCH_COLUMN)

      self.pipeline = beam.Pipeline()
      self.pcoll = (
          self.pipeline
          | 'CreateExamples' >> beam.Create([example])
          | 'ParseExamples' >> beam.Map(lambda x: x.SerializeToString())
          | 'ConvertToRecordBatch' >> converter.BeamSource()
          | 'RunInference' >> run_inference._RunInferenceOnRecordBatch(
                  inference_spec_type, DataType.EXAMPLE))

      self._run_inference_with_beam()


if __name__ == '__main__':
  tf.test.main()
