import numpy as np
import tensorflow as tf
import random

def set_random_seeds(seed):
    """Set random seeds for reproducible results."""

    np.random.seed(seed)

    tf.random.set_seed(seed)

    random.seed(seed)