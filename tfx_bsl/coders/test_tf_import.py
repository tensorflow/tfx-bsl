import tensorflow as tf
import pyarrow as pa

def test_tf_pa():
    print("TF Version:", tf.__version__)
    print("PyArrow Version:", pa.__version__)
