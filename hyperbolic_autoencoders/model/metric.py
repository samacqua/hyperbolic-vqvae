import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def test_metric(*args):
    """Tests custom metric."""
    return 1


def classification_accuracy(loss, data, target, output, aux_model_outputs, aux_loss):
    """Returns the classification accuracy of the model's predictions on a classification task."""
    pred_score, pred_label = torch.max(output, dim=1)
    return (pred_label == target).sum() / target.numel()

def fid(loss, data, target, recon_img, aux_model_outputs, aux_loss):
    """Plots the Fréchet Inception Distance between the original image and the reconstruction."""

    # Renormalize to 0-255 and change to ints.
    n_min, n_max = 0, 255
    r_min, r_max = recon_img.min(), recon_img.max()
    recon_img = ((recon_img - r_min)/(r_max - r_min)*(n_max - n_min) + n_min).type(torch.uint8)

    d_min, d_max = data.min(), data.max()
    data = ((data - d_min) / (d_max - d_min) * (n_max - n_min) + n_min).type(torch.uint8)

    # Make into 3-channels.
    if data.shape[1] == 1:
        data = data.repeat(1, 3, 1, 1)
        recon_img = recon_img.repeat(1, 3, 1, 1)

    # Compute the FID score.
    torch.set_default_tensor_type(torch.FloatTensor)

    fid_model = FrechetInceptionDistance(feature=64)
    fid_model.update(data, real=True)
    fid_model.update(recon_img, real=False)
    score = fid_model.compute()

    torch.set_default_tensor_type(torch.DoubleTensor)

    return score


### VQ-VAE loss ###


def reconstruction_loss(loss, data, target, recon_img, aux_model_outputs, aux_loss):
    """Plots reconstruction component of loss term."""
    reconstruction_loss, codebook_loss, commit_loss, smooth_loss = aux_loss
    return reconstruction_loss


def codebook_loss(loss, data, target, recon_img, aux_model_outputs, aux_loss):
    """Plots codebook loss component of loss term."""
    reconstruction_loss, codebook_loss, commit_loss, smooth_loss = aux_loss
    return codebook_loss


def commit_loss(loss, data, target, recon_img, aux_model_outputs, aux_loss):
    """Plots commit component of loss term."""
    reconstruction_loss, codebook_loss, commit_loss, smooth_loss = aux_loss
    return commit_loss


def smooth_loss(loss, data, target, recon_img, aux_model_outputs, aux_loss):
    """Plots smooth component of loss term."""
    reconstruction_loss, codebook_loss, commit_loss, smooth_loss = aux_loss
    return smooth_loss


def n_active_codes(loss, data, target, recon_img, aux_model_outputs, aux_loss):
    """Plots smooth component of loss term."""
    z_e, emb, argmin, z_e_decoding, z_q_decoding = aux_model_outputs
    return torch.bincount(argmin.view(-1)).count_nonzero()
