"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn

class L1DistanceLoss(nn.Module):
  """Custom L1 loss for distance matrices."""
  def __init__(self, args):
    super(L1DistanceLoss, self).__init__()
    self.args = args
    self.word_pair_dims = (1,2)

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on distance matrices.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the square of the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted distances
      label_batch: A pytorch batch of true distances
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents


class L1DepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self, args):
    super(L1DepthLoss, self).__init__()
    self.args = args
    self.word_dim = 1

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_dim)
      normalized_loss_per_sent = loss_per_sent / length_batch.float()
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents

class RankDepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self, args):
    super(RankDepthLoss, self).__init__()
    self.args = args

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    if total_sents > 0:
      bs, seg_len = predictions_masked.shape
      
      # diff = predictions_masked.view(bs, seg_len, 1) - predictions_masked.view(bs, 1, seg_len)
      # sign = torch.sign(labels_masked.view(bs, seg_len, 1) - labels_masked.view(bs, 1, seg_len))
      
      loss_per_sent = torch.zeros(bs, dtype=predictions_masked.dtype, device=predictions_masked.device)
      for batch_idx in range(bs):
        '''
        predictions_masked_b = predictions_masked[batch_idx]
        labels_masked_b = labels_masked[batch_idx]
        for i in range(length_batch[batch_idx]-1):
          for j in range(i+1, length_batch[batch_idx]):
            loss_per_sent[batch_idx] += 1 - torch.sign(labels_masked_b[i] - labels_masked_b[j])*(predictions_masked_b[i] - predictions_masked_b[j])
            print(torch.sign(labels_masked_b[i] - labels_masked_b[j])*(predictions_masked_b[i] - predictions_masked_b[j]))

        print(loss_per_sent[batch_idx])
        '''
        p = predictions_masked[batch_idx, :length_batch[batch_idx]]
        l = labels_masked[batch_idx, :length_batch[batch_idx]]
        diff = p.view(-1, 1) - p.view(1, -1)
        sign = torch.sign(l.view(-1, 1) - l.view(1, -1))
        loss_per_sent[batch_idx] = 0.5 * torch.sum(1 - diff*sign)
      
      # loss_per_sent = 0.5 * torch.sum(-sign*diff, dim=(1, 2))
      normalized_loss_per_sent = loss_per_sent / length_batch.float()
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
      # print(batch_loss)
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents
