from torch import nn
from torch.nn import functional as F
import torch
from typing import TypeVar
from hypmath import metrics


Tensor = TypeVar('torch.tensor')


def elbo_loss(x: Tensor, recon_x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes the ELBO Optimization objective for gaussian posterior (reconstruction term + regularization term)."""

    reconstruction_function = nn.MSELoss(reduction='sum')
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD


def simple_vq_loss(input_img: Tensor, recon_img: Tensor, encoding: Tensor, quantized_encoding: Tensor,
                   codebook_coef: float = 1., commit_coef: float = 0.5, hyperbolic: bool = True) -> Tensor:
    """Computes a simple loss function to train the VQ-VAE.

    The loss has 3 components.
        reconstruction loss:
        codebook loss:
        commitment loss:

    Args:
        input_img (Tensor): The original data point.
        recon_img (Tensor): The output of the VQ-VAE.
        encoding (Tensor): The encoding before quantization.
        quantized_encoding (Tensor): The quantized encoding of the input.
        codebook_coef (float): The scaling parameter of the codebook component of the loss.
        commit_coef (float): The scaling parameter of the commitment component of the loss.
    """

    reconstruction_loss = F.mse_loss(input_img, recon_img)

    if hyperbolic:
        cb_ = metrics.PoincareDistance(quantized_encoding, encoding.detach())
        codebook_loss = torch.mean(torch.norm(cb_, 2, 1))
        cm_ = metrics.PoincareDistance(quantized_encoding.detach(), encoding)
        commit_loss = torch.mean(torch.norm(cm_, 2, 1))
    else:
        codebook_loss = torch.mean(torch.norm((quantized_encoding - encoding.detach()) ** 2, 2, 1))
        commit_loss = torch.mean(torch.norm((quantized_encoding.detach() - encoding) ** 2, 2, 1))

    return reconstruction_loss + codebook_coef * codebook_loss + commit_coef * commit_loss
