'''
Authors: Geeticka Chauhan, Ruizhi Liao

Custom loss functions for the joint model
'''
import random

import torch
from torch.nn import CosineSimilarity, MarginRankingLoss
from torch.autograd import Variable


def ranking_loss(z_image, z_text, y, report_id, 
                 similarity_function='dot'):
    """
    A custom ranking-based loss function
    Args:
        z_image: a mini-batch of image embedding features (shape=(batch_size, anatomy_size, embedding_size))
        z_text: a mini-batch of text embedding features (shape=(batch_size, anatomy_size, embedding_size))
        y: a mini-batch of image-text labels (shape=(batch_size, anatomy_size, num_classes))
    """
    return imposter_img_loss(z_image, z_text, y, report_id, similarity_function) + \
           imposter_txt_loss(z_image, z_text, y, report_id, similarity_function)

def imposter_img_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the image is an imposter image chosen from the batch 

    Args:
        report_id: a vector of ids coressponding to the dicom ids in MIMIC CXR (shape=(batch_size,))

    """
    loss = torch.zeros(18, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)
    # margin_thresh = 0.5*torch.ones(y.size()[1], device=z_image.device, requires_grad=False)
    for i in range(batch_size):
        # if similarity_function == 'dot':
        # paired_similarity = torch.dot(z_image[i], z_text[i])
        paired_similarity = torch.sum(z_image[i,:,:] * z_text[i,:,:], dim=-1)
        # if similarity_function == 'cosine':
        #     paired_similarity = \
        #         torch.dot(z_image[i], z_text[i])/(torch.norm(z_image[i])*torch.norm(z_text[i]))
        # if similarity_function == 'l2':
        #     paired_similarity = -1*torch.norm(z_image[i]-z_text[i])

        # Select an imposter image index and 
        # compute the maximum margin based on the image label difference
        margin = torch.zeros(1, device=z_image.device, requires_grad=False)
        j = i+1 if i < batch_size - 1 else 0

        if report_id[i] == report_id[j]: 
        # This means the imposter image comes from the same acquisition 
            # margin = 0
            margin = torch.zeros(z_image.size()[1], device=z_image.device, requires_grad=False)
        # elif y[i].item() == -1 or y[j].item() == -1: # '-1' means unlabeled 
        #     margin = 0.5
        else:
            # margin = max(0.5, (y[i] - y[j]).abs().item())
            # margin = torch.maximum(0.5*torch.ones(y.size()[1]), (y[i,:,:] - y[j,:,:]).sum(dim=1).abs())
            diff = (y[i,:,:] - y[j,:,:]).sum(dim=1).abs()
            margin = torch.where(diff > 0.5, diff, 0.5)

        # if similarity_function == 'dot':
        # imposter_similarity = torch.dot(z_image[j], z_text[i])
        imposter_similarity = torch.sum(z_image[j,:,:] * z_text[i,:,:], dim=-1)
        # if similarity_function == 'cosine':
        #     imposter_similarity = \
        #         torch.dot(z_image[j], z_text[i])/(torch.norm(z_image[j])*torch.norm(z_text[i]))
        # if similarity_function == 'l2':
        #     imposter_similarity = -1*torch.norm(z_image[j]-z_text[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        # if diff_similarity > 0:
        # loss = loss + diff_similarity
        loss = torch.where(diff_similarity > 0, loss + diff_similarity, loss)

    return loss / batch_size # 'mean' reduction

def imposter_txt_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the text is an imposter text chosen from the batch 
    """
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)
    # margin_thresh = 0.5*torch.ones(y.size()[1], device=z_image.device, requires_grad=False)
    for i in range(batch_size):
        # if similarity_function == 'dot':
            # paired_similarity = torch.dot(z_image[i], z_text[i])
        paired_similarity = torch.sum(z_image[i,:,:] * z_text[i,:,:], dim=-1)
        # if similarity_function == 'cosine':
        #     paired_similarity = \
        #         torch.dot(z_image[i], z_text[i])/(torch.norm(z_image[i])*torch.norm(z_text[i]))
        # if similarity_function == 'l2':
        #     paired_similarity = -1*torch.norm(z_image[i]-z_text[i])

        # Select an imposter text index and 
        # compute the maximum margin based on the image label difference
        j = i+1 if i < batch_size - 1 else 0
        if report_id[i] == report_id[j]: 
            # This means the imposter report comes from the same acquisition 
            # margin = 0
            margin = torch.zeros(z_image.size()[1], device=z_image.device, requires_grad=False)
        # elif y[i].item() == -1 or y[j].item() == -1: # '-1' means unlabeled
        #     margin = 0.5
        else:
            # margin = max(0.5, (y[i] - y[j]).abs().item())
            # margin = torch.maximum(margin_thresh, (y[i,:,:] - y[j,:,:]).sum(dim=1).abs())
            diff = (y[i,:,:] - y[j,:,:]).sum(dim=1).abs()
            margin = torch.where(diff > 0.5, diff, 0.5)

        # if similarity_function == 'dot':
        #     imposter_similarity = torch.dot(z_text[j], z_image[i])
        imposter_similarity = torch.sum(z_text[j,:,:] * z_image[i,:,:], dim=-1)
        # if similarity_function == 'cosine':
        #     imposter_similarity = \
        #         torch.dot(z_text[j], z_image[i])/(torch.norm(z_text[j])*torch.norm(z_image[i]))
        # if similarity_function == 'l2':
        #     imposter_similarity = -1*torch.norm(z_text[j]-z_image[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        # if diff_similarity > 0:
        # loss = loss + diff_similarity
        loss = torch.where(diff_similarity > 0, loss + diff_similarity, loss)

    return loss / batch_size # 'mean' reduction

def dot_product_loss(z_image, z_text):
    batch_size = z_image.size(0)
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    for i in range(batch_size):
        loss = loss - torch.dot(z_image[i], z_text[i])
    return loss / batch_size

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
