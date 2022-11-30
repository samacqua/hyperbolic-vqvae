import torch
import pytorch_fid


def test_metric(*args):
    """Tests custom metric."""
    return 1


def fid(loss, data, target, recon_img, aux_model_outputs, aux_loss):
    """Plots the Fréchet Inception Distance between the original image and the reconstruction."""
    raise NotImplementedError


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

