import pyarrow as pa
import tensorflow as tf

def test_tf():
    print("TF Version:", tf.__version__)
    print("PyArrow Version:", pa.__version__)
