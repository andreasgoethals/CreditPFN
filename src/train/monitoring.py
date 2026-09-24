"""Fixed, disjoint context/validation/query rows for descriptive monitoring."""
import numpy as np
from sklearn.model_selection import train_test_split


def monitor_indices(y, task_type, query_fraction, seed):
    indices = np.arange(len(y))
    if len(indices) < 6:
        raise ValueError("Monitoring requires at least six rows")

    def split(rows, fraction, state):
        stratify = np.asarray(y)[rows] if task_type == "classification" else None
        try:
            return train_test_split(rows, test_size=fraction, random_state=state, stratify=stratify)
        except ValueError:
            return train_test_split(rows, test_size=fraction, random_state=state)

    development, query = split(indices, query_fraction, seed)
    context, validation = split(development, .2, (seed+1) % (2**32-1))
    return context, validation, query
