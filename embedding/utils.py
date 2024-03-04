import numpy as np


def normalize_embedding(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, ord=2)
    normalized_embeddings = embeddings / norms[:, np.newaxis]
    return normalized_embeddings


def average_pooling(embeddings, attention_masks):
    attention_mask_bools = attention_masks[..., np.newaxis].astype(bool)
    embeddings = np.where(~attention_mask_bools, 0.0, embeddings)
    return np.sum(embeddings, axis=1) / np.sum(attention_masks, axis=1)[..., np.newaxis]
