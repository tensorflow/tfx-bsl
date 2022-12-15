# TFXIO's schema interpretation

[TOC]

Data parsing by TFXIO sources is configured by
[Schema](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L72).
Among other things, the schema is converted to a "feature spec" (sometimes
called a "parsing spec") which is a dict whose keys are feature names and values
are one of:

*   [`tf.io.FixedLenFeature`](https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature)
*   [`tf.io.VarLenFeature`](https://www.tensorflow.org/api_docs/python/tf/io/VarLenFeature)
*   [`tf.io.SparseFeature`](https://www.tensorflow.org/api_docs/python/tf/io/SparseFeature)
*   [`tf.io.RaggedFeature`](https://www.tensorflow.org/api_docs/python/tf/io/RaggedFeature)

Feature spec is conventionally used by
[`tf.io.parse_example`](https://www.tensorflow.org/api_docs/python/tf/io/parse_example)
and
[`tf.io.parse_sequence_example`](https://www.tensorflow.org/api_docs/python/tf/io/parse_sequence_example)
to convert from serialized
[`Example`](https://www.tensorflow.org/api_docs/python/tf/train/Example) and
[`SequenceExample`](https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample)
protos to batched tensors. This doc describes how the conversion and parsing are
performed.

Although TFXIOs don't typically use the TensorFlow ops for parsing in a Beam
pipeline and the data format is not restricted to the two proto types, the
result is always consistent with that of the parsing ops. Therefore, we express
the schema interpretation logic in terms of feature specs that essentially
describe how tensors are constructed from primitives.

Key objects of the schema that can be used to describe parsing are
[`Feature`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L74),
[`SparseFeature`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L77),
and
[`TensorRepresentation`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L131).
Normally, only the first two are set by TFXIO users to control the produced
tensors, but the last one can also be used for a more fine-grained control. In
the following sections we will describe how these are mapped to feature spec and
give some examples of advanced usage.

## Feature

### Primitive types

<!-- BEGIN GOOGLE-INTERNAL 

Schema interpretation logic has two versions controlled by
[`generate_legacy_feature_spec`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L115):

<section class="zippy open">

`generate_legacy_feature_spec: false`  END GOOGLE-INTERNAL -->

1.  If the feature has a fixed
    [`shape`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L165)
    (i.e. all dimensions have non-negative integer sizes), then the feature must
    always be present (`presence.min_fraction` == 1.0), and a
    `tf.io.FixedLenFeature` with the given shape and type will be produced for
    it.
1.  Otherwise, a `tf.io.VarLenFeature` of the corresponding type will be
    produced.

<!-- BEGIN GOOGLE-INTERNAL 

</section>

<section class="zippy open">

`generate_legacy_feature_spec: true` (default)

[`value_count`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L168)
and
[`presence`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L155)
are used to determine kind of the spec and its shape according to the following
table:

value_count                 | presence.min_fraction | feature spec
--------------------------- | --------------------- | ------------
min = max = 1               | 1                     | `tf.io.FixedLenFeature(`<br> &nbsp;&nbsp;&nbsp;&nbsp;`shape=(),`<br> &nbsp;&nbsp;&nbsp;&nbsp;`dtype=<TF type>,`<br> &nbsp;&nbsp;&nbsp;&nbsp;`default_value=None)`
min = max = 1               | <1                    | `tf.io.FixedLenFeature(`<br> &nbsp;&nbsp;&nbsp;&nbsp;`shape=(),`<br> &nbsp;&nbsp;&nbsp;&nbsp;`dtype=<TF type>,`<br> &nbsp;&nbsp;&nbsp;&nbsp;`default_value=<default>)`
min = max = k > 1           | 1                     | `tf.io.FixedLenFeature(`<br> &nbsp;&nbsp;&nbsp;&nbsp;`shape=(k,),`<br>&nbsp;&nbsp;&nbsp;&nbsp; `dtype=<TF type>,`<br> &nbsp;&nbsp;&nbsp;&nbsp;`default_value=None)`<br>
min = max = k > 1           | <1                    | `tf.io.FixedLenFeature(`<br> &nbsp;&nbsp;&nbsp;&nbsp;`shape=(k,),`<br> &nbsp;&nbsp;&nbsp;&nbsp;`dtype=<TF type>,`<br> &nbsp;&nbsp;&nbsp;&nbsp;`default_value=<default>)`
min = max = 0 or min != max | any                   | `tf.io.VarLenFeature(`<br> &nbsp;&nbsp;&nbsp;&nbsp;`dtype=<TF type>)`

</section>
 END GOOGLE-INTERNAL -->

Where TensorFlow types and default values are derived according to the following
table:

Schema type | TF type    | Default value
----------- | ---------- | -------------
BYTES       | tf.string  | b''
INT         | tf.int64   | -1
FLOAT       | tf.float32 | -1.0

### `STRUCT` type

`STRUCT` features are typically used for `SequenceExample` inputs. At the moment
only one level of nestedness is supported and leaf features from
[`StructDomain`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L194)
of the feature result into top-level `tf.io.RaggedFeature`s.

## SparseFeature

Results in `tf.io.SparseFeature` with the given value and index columns. Dense
shape is derived from index columns' `int_domain`s.

`SparseFeature`s are only supported with `generate_legacy_feature_spec: false`.

## Advanced usage

TFXIO uses
[`TensorRepresentations`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L763)
to infer parsing spec from the schema `Feature`s and `SparseFeature`s. Manually
specifying `TensorRepresentations` in the schena allows a more refined control
of the input data representation. Here are a few examples:

### Parsing variable length feature as ragged tensor

```python
feature {
  name: "varlen"
  type: BYTES
}
tensor_representation_group {
  key: ""
  value {
    tensor_representation {
      key: "varlen"
      value {
        ragged_tensor {
          feature_path { step: "varlen" }
          row_partition_dtype: INT64
        }
      }
    }
  }
}
```

Produced feature spec:

```python
{
  "varlen":
      tf.io.RaggedFeature(
          value_key="varlen",
          dtype=tf.string,
          partitions=(,),
          row_splits_dtype=tf.int64)
}
```

### Parsing as a nested ragged tensor

```python
feature {
  name: "value"
  type: BYTES
}
feature {
  name: "row_length"
  type: INT
}
tensor_representation_group {
  key: ""
  value {
    tensor_representation {
      key: "ragged"
      value {
        ragged_tensor {
          feature_path { step: "value" }
          partition { row_length: "row_length" }
          row_partition_dtype: INT64
        }
      }
    }
  }
}
```

Produced feature spec:

```python
{
  "ragged":
      tf.io.RaggedFeature(
          value_key="value",
          dtype=tf.string,
          partitions=(tf.io.RaggedFeature.RowLengths("row_length"),),
          row_splits_dtype=tf.int64)
}
```

### Parsing as a sparse tensor

```python
feature {
  name: "value"
  type: FLOAT
}
feature {
  name: "index0"
  type: INT
}
feature {
  name: "index1"
  type: INT
}
tensor_representation_group {
  key: ""
  value {
    tensor_representation {
      key: "sparse"
      value {
        sparse_tensor {
          index_column_names: ["index0", "index1"]
          value_column_name: "value"
          dense_shape {
            dim {
              size: 10
            }
            dim {
              size: 20
            }
          }
          already_sorted: true
        }
      }
    }
  }
}
```

Produced feature spec:

```python
{
  "sparse":
      tf.io.SparseFeature(
          index_key=["index0", "index1"],
          value_key="value",
          dtype=tf.float32,
          size=[10, 20],
          already_sorted=True)
}
```

### Parsing `STRUCT` feature

One of the applications of `STRUCT` features is `SequenceExample` reading and
parsing. In this case context features should be declared on the top level while
sequence features - in a `struct_domain` of a feature named
["##SEQUENCE##"](https://github.com/tensorflow/tfx-bsl/blob/master/tfx_bsl/tfxio/tf_sequence_example_record.py#L35):

```python
feature {
  name: "##SEQUENCE##"
  type: STRUCT
  struct_domain {
    feature {
      name: "seq_int_feature"
      type: INT
      value_count {
        min: 0
        max: 2
      }
    }
    feature {
      name: "seq_string_feature"
      type: BYTES
      value_count {
        min: 0
        max: 2
      }
    }
  }
}
tensor_representation_group {
  key: ""
  value {
    tensor_representation {
      key: "seq_string_feature"
      value {
        ragged_tensor {
          feature_path {
            step: "##SEQUENCE##" step: "seq_string_feature"
          }
        }
      }
    }
    tensor_representation {
      key: "seq_int_feature"
      value {
        ragged_tensor {
          feature_path {
            step: "##SEQUENCE##" step: "seq_int_feature"
          }
        }
      }
    }
  }
}
```

Resulting parsing spec:

```python
{
  "seq_string_feature":
      tf.io.RaggedFeature(
          dtype=tf.string,
          value_key="seq_string_feature",
          row_splits_dtype=tf.int64,
          partitions=[]),
  "seq_int_feature":
      tf.io.RaggedFeature(
          dtype=tf.int64,
          value_key="seq_int_feature",
          row_splits_dtype=tf.int64,
          partitions=[])
}
```
