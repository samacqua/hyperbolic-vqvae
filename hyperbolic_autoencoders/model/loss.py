from torch import nn
from torch.nn import functional as F
import torch
from typing import TypeVar, Tuple
from hypmath import metrics

Tensor = TypeVar('torch.tensor')


def elbo_loss(x: Tensor, recon_x: Tensor, mu: Tensor, logvar: Tensor, **kwargs) -> Tensor:
    """Computes the ELBO Optimization objective for gaussian posterior (reconstruction term + regularization term)."""

    reconstruction_function = nn.MSELoss(reduction='sum')
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD


def vq_loss(input_img: Tensor, recon_img: Tensor, z_e: Tensor, z_q: Tensor, argmin: Tensor,
            emb_decoded: Tensor, quantized_decoded: Tensor, *args,
            beta: float = 1., alpha: float = 0.5, hyperbolic: bool = False,
            bounded_measure: bool = False, enforce_smooth: bool = False, smooth_coef: float = 1., **kwargs
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Computes the loss function to train the VQ-VAE.

    The loss has 3-4 components.
        reconstruction loss: loss propagated through encoder and decoder from difference between reconstructed image and
            original image. Uses "straight-through" assumption, where quantization operation is ignored in terms of how
            it would effect gradient.
        codebook loss: Distance between embedding and quantized embedding. Stop gradient applied to embedding so the
            loss only effects the codebooks.
        commitment loss: Distance between embedding and quantized embedding. Stop gradient applied to codebook so the
            loss only effects the encoder (makes it "commit" to a codebook vector).
        smooth loss: Distance between reconstruction from embedding and reconstruction from quantized embedding. Only
            effect decoder.

    Args:
        input_img (Tensor): The original data point.
        recon_img (Tensor): The output of the VQ-VAE, with gradients only going through chosen codebook vectors (stop
            gradient).
        z_e (Tensor): The encoding before quantization.
        z_q (Tensor): The quantized encoding of the input with gradients only going through the codebook
            (does not include encoder).
        argmin (Tensor): The index of the closest codebook vector.
        emb_decoded (Tensor): The output of decoding the un-quantized embedding. Gradients only flow through decoder.
        quantized_decoded (Tensor): The output of decoding the quantized embedding, only carrying gradients through the
            decoder.
        beta (float): The scaling parameter of the codebook component of the loss.
        alpha (float): The scaling parameter of the commitment component of the loss.
        hyperbolic (bool): Boolean to compute the loss in hyperbolic space for the embeddings. If True, then the
            distance functions will use hyperbolic equivalents.
        bounded_measure (bool): Boolean to compute the distance using a bounded measure in Euclidean space. So, instead
            of computing Euclidean distance, computes the cosine similarity. In hyperbolic space, instead of computing
            hyperbolic distance, calculates the Minkowski inner product (which is not bounded, but is the hyperbolic
            analog of cosine similarity as per
            https://math.stackexchange.com/questions/2852458/bounded-similarity-measure-for-points-in-hyperbolic-space).
        enforce_smooth (bool): Boolean to include a loss component to enforce the decoder is smooth.
        smooth_coef (float): The scaling parameter of the smoothness component of the loss.
    """

    # The straight-through loss. Weights only flow through codebooks that were selected.
    reconstruction_loss = F.mse_loss(input_img, recon_img)

    # The commitment loss. Depends on if the embeddings are hyperbolic, and what distance metric is used.
    if hyperbolic:

        if bounded_measure:
            raise NotImplementedError
        # Distance on Poincare Ball.
        else:
            cb_ = metrics.PoincareDistance(z_q, z_e.detach(), dim=1)
            codebook_loss = torch.mean(cb_)
            cm_ = metrics.PoincareDistance(z_q.detach(), z_e, dim=1)
            commit_loss = torch.mean(cm_)
    else:
        if bounded_measure:
            raise NotImplementedError
        # Euclidean distance.
        else:
            codebook_loss = torch.mean(torch.norm((z_q - z_e.detach()), p='fro', dim=1))
            commit_loss = torch.mean(torch.norm((z_q.detach() - z_e), p='fro', dim=1))

    # The smoothness loss.
    smooth_loss = 0
    if enforce_smooth:
        smooth_loss = torch.norm(emb_decoded - quantized_decoded)

    vq_loss = reconstruction_loss + codebook_loss * beta + commit_loss * alpha + smooth_loss * smooth_coef

    return vq_loss, reconstruction_loss, codebook_loss, commit_loss, smooth_loss
