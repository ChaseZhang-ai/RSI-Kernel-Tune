"""Embedding lookup reference: PyTorch, an exact gather."""

import torch


def run(table, ids):
    return table[ids]
