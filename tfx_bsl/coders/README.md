# Standardized TFX inputs decoder specifications

This document contains specifications of `tfx_bsl` decoders (which decodes
various data format to `pyarrow.RecordBatch`. They are considered contracts to
consumers of the record batches (e.g. TFX libraries as well as custom TFX
components that needs to read the data using
[Standardized TFX inputs](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md))
and are subject to semantic versioning.

This document can also be used as a guide for custom decoders (which may be
part of a custom `TFXIO` implementation) -- the Standardized TFX inputs project
guarantees that as long as the schema of the Apache Arrow record batches is
consistent with the specification of one of the documented decoders, then any
TFX component that adopts the Standardized TFX inputs should be able to process
the output of the custom decoder.

## Notions

In below sections, we will use Apache Arrow's type factory to denote the type of
a column in an Arrow schema, for example:

```python
pa.large_list(pa.int64())
```

means a `LargeListType` whose child array type is an `Int64ArrayType`.

We will also use `primitives` to denote the set of `{pa.int64(), pa.float32(),
pa.large_binary()}`.

## `tf.Example` Decoder

This decoder has two working modes -- schema-ful mode and schema-less mode.
Here "schema" means the TensorFlow Metadata (TFMD) [`Schema`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto).

### Schema-ful mode

Under this mode, the schema of the output record batches is fully determined by
the TFMD `Schema` (i.e. it does not depend on the data). Each `Feature` in
`Schema.feature` maps to one column in the output record batch. The order of
columns is the same as the order of the `Feature`s in `Schema.feature`.

Many TFX components require a TFMD `Schema` when dealing with `tf.Examples`
because they need to know the Arrow schema of the data before seeing the data.

TFMD `Schema.feature.type` | Decoded Type
-------------------------- | ----------------------------------
`BYTES`                    | `pa.large_list(pa.large_binary())`
`INT`                      | `pa.large_list(pa.int64())`
`FLOAT`                    | `pa.large_list(pa.float32())`

Types not listed in the above table are invalid to this decoder.

#### Corner cases

*   If a `tf.Example` contains a feature whose name is not listed in the TFMD
    `Schema`, the feature will be ignored.
*   If a `tf.Example` contains a feature whose "kind" `oneof` conflicts with the
    type specified in the TFMD `Schema`, for example, if `bytes_list` is set but
    the TFMD `Schema` says `INT`, an error will be raised.

### Schema-less mode

Under this mode, the schema of the output record batch depends on the input
batch of `tf.Example`s. Different batches may result in different Arrow schemas.

The columns in the output schema is a union of all the features seen in the
input batch of `tf.Example`s. The columns are sorted by their names in
lexicographical order.

If a feature `"f"` is present in one `tf.Example` in the input batch, then for
any other `tf.Example`s in the batch, one of the following must hold:

*   it does not contain `"f"`.
*   it contains `"f"`, and it either does not have `tf.Feature.kind` set, or the
    same {`bytes_list`,`int64_list`,`float_list`} is set.

An error will be raised if any `tf.Example` in the batch violates the above.

It is allowed that all the instances of `"f"` in the input batch of
`tf.Example`s to not have the "kind" `oneof` set.

| `tf.Feature.kind` of feature `"f"` in | Decoded Type                       |
: at least one `tf.Example` in input    :                                    :
| ------------------------------------- | ---------------------------------- |
| `bytes_list`                          | `pa.large_list(pa.large_binary())` |
| `int64_list`                          | `pa.large_list(pa.int64())`        |
| `float_list`                          | `pa.large_list(pa.float32())`      |
| no `tf.Example` has `tf.Feature.kind` | `pa.null()`                        |
: set                                   :                                    :

Also see ["nuance regarding `null`"](#null-nuances).

Note: this mode makes it possible for the `TFXIO` implementations for
the `tf.Example` format to take no TFMD `Schema`, however, without the TFMD
`Schema`, such `TFXIO` instances have limited functionalities and may not work
with some TFX components.


### Nuances regarding `null` {#null-nuances}

The decoded Arrow columns may contain `null`s. And a `null` element in a
`pa.large_list(primitives)` array is different from an empty list (`[]`)
element.

A `null` element in a column means one of the following:

*   The `tf.Feature` does not have the "kind" `oneof` set (i.e. it contains none
    of `bytes_list`, `int64_list`, `float_list`).
*   The `tf.Example` is missing the feature named after the column.

An empty list element means that `{bytes_list,int64_list,float_list}.value` is
empty.

Note that under the schema-less mode, if none of the `tf.Feature` for feature
`f` has the "kind" `oneof` set (but the feature is present in some of the
examples), a `pa.null()` array will be produced for `f`, indicating that the
type of the feature is unknown, but all the contents are `null`s (tip: imagine a
`pa.null()` to be a `pa.large_list<?>`, with all elements being `null`s).

## `tf.SequenceExample` Decoder

This coder also has schema-ful and schema-less working modes. The specs and
treatments regarding `context` features are exactly the same as those for
`tf.Example` documented above. Thus this section only focuses on `feature_lists`
(or "sequence features").

Note: `tf.Example` and `tf.SequenceExample` are wire-format compatible -- the
wire bytes of one can be parsed as an instance of the other. Of course,
`feature_lists` will be discarded or set to empty in such processes.

### Schema-ful mode

Under this mode, all the sequence features are expected to be contained in a
"struct feature" (a TFMD
[`Feature`](https://github.com/tensorflow/metadata/blob/f01d580217c1c557a06ea21709422ca290797797/tensorflow_metadata/proto/v0/schema.proto#L114)
of type
[`STRUCT`](https://github.com/tensorflow/metadata/blob/f01d580217c1c557a06ea21709422ca290797797/tensorflow_metadata/proto/v0/schema.proto#L643),
and it contains a
[`StructDomain`](https://github.com/tensorflow/metadata/blob/f01d580217c1c557a06ea21709422ca290797797/tensorflow_metadata/proto/v0/schema.proto#L510)
which has `Feature`s nested under), named `"##SEQUENCE##"`. Each `Feature` in
`"##SEQUENCE##"` feature's `struct_domain` maps to one sequence feature to be
decoded. Those `Feature`s and their decoded types are in the table below:

TFMD `Schema.feature.type` | Decoded Type
-------------------------- | ----------------------------------
`BYTES`                    | `pa.large_list(pa.large_list(pa.large_binary()))`
`INT`                      | `pa.large_list(pa.large_list(pa.int64()))`
`FLOAT`                    | `pa.large_list(pa.large_list(pa.float32()))`

Here is an example of a TFMD schema with one context feature and two sequence
features:

```
feature {
  name: "context_1"
  type: INT
}
feature {
  name: "##SEQUENCE##"
  type: STRUCT
  struct_domain {
    feature {
      name: "sequence_1"
      type: INT
    }
    feature {
      name: "sequence_2"
      type: BYTES
    }
  }
}
```

The result record batch will contain a column named `"##SEQUENCE##"`, of type
`pa.struct(...)`. That column will have child arrays corresponding to sequence
features, of types described in the table above. The ordering of childs in the
`"##SEQUENCE##"` column follows the ordering of the sequence features described
in the schema. The `##SEQUENCE##` column, if exists, will always be the last
column in a record batch.

Here is the Arrow schema of record batches that would be produced by decoding
with the TFMD Schema given above:

```
column "context_1"
  type: large_list<int64>

column "##SEQUENCE##"
  type: struct<[("sequence_1", large_list<large_list<int64>>),
                ("sequence_2", large_list<large_list<large_binary>>)]
    child "sequence_1"
      type: large_list<large_list<int64>>
    child "sequence_2"
      type: large_list<large_list<large_binary>>
```

#### Corner cases

*   A context feature and a sequence feature may have the same name, as allowed
    by the TFMD schema.
*   `"##SEQUENCE##"` is a reserved name.
*   If a schema does not have the `"##SEQUENCE##"` feature, the decoder
    essentialy degrades to a `tf.Example` decoder.
*   Also see the corner cases for the `tf.Example` decoder.

### Schema-less mode

All the points mentioned in the `tf.Example` decoder spec hold, and extend to
sequence features, plus the following:

*   Similar to the schema-ful case, there may be a `"##SEQUENCE##"` column (
    again, always the last column) in the result record batch that groups all
    the sequence features.
*   If the type of a sequence feature can not be determined, it will be decoded
    as a `pa.large_list(pa.null())` .
*   The decoded value of a sequence feature will be `null` if the sequence
    feature is missing from the `feature_lists` map, but appears in at least one
    another example in the batch.(note that this is referring to the outer level
    `large_list`)
*   The decoded value of an episode of a sequence feature value will be `null`
    if the `tf.Feature` does not have the "kind" `oneof` set.
*   If none of the `SequenceExample`s in a batch has `feature_lists`, then
    the `"##SEQUENCE##"` column will not be produced.
