from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

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
from tfx_bsl.public.beam import run_inference
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
          | 'RunInference' >> run_inference.RunInference(
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


if __name__ == '__main__':
  tf.test.main()
