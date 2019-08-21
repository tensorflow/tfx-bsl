# TFX Basic Shared Libraries

TFX Basic Shared Libraries (`tfx_bsl`) contains libraries shared by many
[TensorFlow eXtended (TFX)](https://www.tensorflow.org/tfx) components.

This package is __not__ intended for direct use by TFX users, and its APIs
should be considered internal to TFX (therefore there is no backward or forward
compatibility guarantee) unless otherwise remarked.

Each minor version of a TFX component or TFX itself, if it needs to
depend on `tfx_bsl`, will depend on a specific minor version of it (e.g.
`tensorflow_data_validation` 0.14.\* will depend on and only work with `tfx_bsl`
0.14.\*)
