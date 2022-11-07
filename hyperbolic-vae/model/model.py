import torch
from base import BaseModel
from torch import nn
from torch.nn import functional as F
from hypmath import WrappedNormal, poincareball, metrics
from torch.distributions import Normal
from typing import TypeVar, List, Optional, Tuple
from geoopt.manifolds import Lorentz
from .nearest_embed import NearestEmbed, nearest_embed

Tensor = TypeVar('torch.tensor')


class Encoder(BaseModel):
    """Convolutional encoder. Sequence of convolutional layer + batch norm + lReLU."""

    def __init__(self, hidden_dims: List[int], in_channels: int):
        """Initializes the encoder.

        Params:
            hidden_dims (List[int]): List of the depth of each convolutional layer.
            in_channels: (int): The number of channels in the input image.
        """
        super(Encoder, self).__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=(3, 3), stride=(2, 2), padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(BaseModel):
    """Convolutional decoder. Sequence of convolutional transpose layer + batch norm + lReLU."""

    def __init__(self, hidden_dims: List[int]):
        """Initializes the decoder.

        Params:
            hidden_dims (List[int]): List of the depth of each convolutional layer.
        """

        super(Decoder, self).__init__()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), output_padding=(1, 1)),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class VanillaVAE(BaseModel):

    def __init__(self,
                 in_channels: int,
                 latent_dims: int,
                 img_size: int,
                 hyperbolic: bool = True,
                 hidden_dims: List[int] = None,
                 out_channels: Optional[int] = None) -> None:
        """Instantiates the VAE model.

        For shape comments, B = batch size, C = number of channels in the input image, H = height, W = width, E =
        number of channels returned from the encoder, D = latent dimension, E1 = depth of first convolution of encoder
        & last convolution of decoder, C2 = number of channels in the output image.

        Params:
            in_channels (int): Number of input channels.
            latent_dims (int): Size of latent dimensions.
            img_size (int): The height and width of the input image.
            hyperbolic (bool): Controls if embedding in hyperbolic space.
            hidden_dims (List[int]): List of hidden dimensions, reversed for decoder. Defaults to
                [32, 64, 128, 256, 512]. So, the encoder will have 5 layers ending with depth=512. This final encoding
                will be used to generate the encoding of the latent gaussians, then the decoder takes these encodings
                and will have 5 layers ([512, 256, 128, 64, 32]) to create a "decoding", where a final layer will
                convert to original image shape + depth.
            out_channels (int): Number of output channels. Defaults to number of input channels.
        """
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dims
        self.manifold = poincareball.PoincareBall(self.latent_dim) if hyperbolic else None
        self.hyperbolic = hyperbolic

        # Check images in correct format.
        hidden_dims = hidden_dims or [32, 64, 128, 256, 512]

        def is_power_of_two(n: int) -> bool:
            """Checks if int is power of 2. https://stackoverflow.com/a/57027610."""
            return (n != 0) and (n & (n - 1) == 0)

        assert is_power_of_two(img_size), "Can currently only support images with H = W = power of 2 for simplicity."
        self.hw_prime = int(img_size * 2 ** -len(hidden_dims))
        assert self.hw_prime, "Too much conv for image size. Either decrease num hidden layers or increase img size."
        self.img_size = img_size

        # Create encoder.
        self.hidden_dims = hidden_dims.copy()
        self.encoder = Encoder(hidden_dims, in_channels)

        # Parameterization of latent gaussians.
        # If you flatten the encoding (B x E x H' x W') becomes (B x (E * H' * W')) = (B x D').
        d_prime = self.hw_prime ** 2
        self.fc_mu = nn.Linear(hidden_dims[-1] * d_prime, latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1] * d_prime, latent_dims)

        # Build decoder base.
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * d_prime)
        hidden_dims.reverse()
        self.decoder = Decoder(hidden_dims)

        # Build the final layer to recreate the image.
        out_channels = out_channels or in_channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=(3, 3), padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encodes the input by passing through the convolutional network
        and outputs the latent variables.

        Params:
            input (Tensor): Input tensor [B x C x H x W]

        Returns:
            mu (Tensor) and log_var (Tensor) of latent variables
        """

        result = self.encoder(input)    # B x E x H' x W'
        result = torch.flatten(result, start_dim=1)     # B x (E * H' * W')

        # Split the result into mu and var components of the latent Gaussian distribution.
        mu = self.fc_mu(result)          # B x D
        log_var = self.fc_var(result)     # B x D
        if self.hyperbolic:
            mu = self.manifold.expmap0(mu)   # B x D
            log_var = F.softplus(log_var)     # B x D

        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent variables
        onto the image space.

        Params:
            z (Tensor): Latent variable [B x D]

        Returns:
            result (Tensor) [B x C x H x W]
        """

        if self.hyperbolic:
            z = self.manifold.logmap0(z)    # B x D
        result = self.decoder_input(z)  # B x (E1 * H' * W')
        result = result.view(-1, self.hidden_dims[-1], self.hw_prime, self.hw_prime)     # B x E x H' x W'
        result = self.decoder(result)   # B x E1 x H'' x W''
        result = self.final_layer(result)   # B x C2 x H x W

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1)

        Params:
            mu (Tensor): Mean of Gaussian latent variables [B x D]
            logvar (Tensor): log-Variance of Gaussian latent variables [B x D]

        Returns: 
            z (Tensor) [B x D]
        """

        std = torch.exp(0.5 * logvar)
        dist = WrappedNormal(mu, std, self.manifold) if self.hyperbolic else Normal(mu, std)
        z = dist.rsample()

        return z

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the model.

        Args:
            x (Tensor) [B x C x H x W]

        Returns:
            A three-tuple of the reconstructed image, the mean of the encoding gaussian, and the log variance.
        """

        mu, log_var = self.encode(x)    # (B x D, B x D)
        z = self.reparameterize(mu, log_var)    # B x D
        output = self.decode(z)     # B x C x H x W

        return output, mu, log_var

    def sample(self,
               num_samples: int,
               current_device: str) -> Tensor:
        """
        Samples from the latent space and return the corresponding image space map.

        Params:
            num_samples (Int): Number of samples
            current_device (Int): Device to run the model

        Returns:
            samples (Tensor)
        """

        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image

        Params:
            x (Tensor): input image Tensor [B x C x H x W]

        Returns:
            (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VQVAE(BaseModel):
    """Hyperbolic Vector Quanitized Variational Autoencoder"""

    def __init__(self,
                 in_channels: int,
                 latent_dims: int,
                 img_size: int,
                 n_codebook: int,
                 hyperbolic: bool = True,
                 hidden_dims: List[int] = None,
                 out_channels: Optional[int] = None) -> None:
        """Instantiates the VAE model.

        For shape comments, B = batch size, C = number of channels in the input image, H = height, W = width, E =
        number of channels returned from the encoder, D = latent dimension, E1 = depth of first convolution of encoder
        & last convolution of decoder, C2 = number of channels in the output image.

        Params:
            in_channels (int): Number of input channels.
            latent_dims (int): Size of latent dimensions.
            img_size (int): The height and width of the input image.
            n_codebook (int): The number of codebook vectors.
            hidden_dims (List[int]): List of hidden dimensions, reversed for decoder. Defaults to
                [32, 64, 128, 256, 512]. So, the encoder will have 5 layers ending with depth=512. This final encoding
                will be used to generate the encoding of the latent gaussians, then the decoder takes these encodings
                and will have 5 layers ([512, 256, 128, 64, 32]) to create a "decoding", where a final layer will
                convert to original image shape + depth.
            out_channels (int): Number of output channels. Defaults to number of input channels.
        """
        super(VQVAE, self).__init__()

        self.latent_dim = latent_dims
        self.manifold = poincareball.PoincareBall(self.latent_dim)
        self.hyperbolic = hyperbolic
        out_channels = out_channels or in_channels

        # Check images in correct format.
        hidden_dims = hidden_dims or [32, 64, 128, 256, 512]

        def is_power_of_two(n: int) -> bool:
            """Checks if int is power of 2. https://stackoverflow.com/a/57027610."""
            return (n != 0) and (n & (n - 1) == 0)

        assert is_power_of_two(img_size), "Can currently only support images with H = W = power of 2 for simplicity."
        self.hw_prime = int(img_size * 2 ** -len(hidden_dims))
        assert self.hw_prime, "Too much conv for image size. Either decrease num hidden layers or increase img size."

        # Create encoder.
        self.hidden_dims = hidden_dims.copy()
        self.encoder = Encoder(hidden_dims, in_channels)

        # Parameterization of latent gaussians.
        # If you flatten the encoding (B x E x H' x W') becomes (B x (E * H' * W')) = (B x D').
        d_prime = self.hw_prime ** 2
        self.fc_embed = nn.Linear(hidden_dims[-1] * d_prime, latent_dims)

        d = 64
        bn = True
        self.full_encoder = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )
        self.full_decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, out_channels, kernel_size=4, stride=2, padding=1),
        )

        # Build Quantization.
        self.emb = NearestEmbed(n_codebook, latent_dims, hyperbolic=hyperbolic)

        # Build decoder base.
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * d_prime)
        hidden_dims.reverse()
        self.decoder = Decoder(hidden_dims)

        # Build the final layer to recreate the image.
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=(3, 3), padding=1),
            nn.Tanh())

    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the model.

        Args:
            x (Tensor) [B x C x H x W]

        Returns:
            A three-tuple of the reconstructed image, the unquantized embedding, and the quantized embedding.
        """

        # TODO: when to detatch for loss?

        z_e = self.encode(x)
        z_q, z_argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())

        return self.decode(z_q), z_e, emb

    def encode(self, x: Tensor) -> Tensor:
        """
        Encodes the input by passing through the convolutional network
        and outputs the latent variables.

        Params:
            x (Tensor): Input tensor [B x C x H x W]

        Returns:
            embed (Tensor): The hyperbolic embedding of the input.
        """

        # result = self.encoder(x)    # B x E x H' x W'
        # result = torch.flatten(result, start_dim=1)     # B x (E * H' * W')
        # embed = self.fc_embed(result)          # B x D
        embed = self.full_encoder(x)
        if self.hyperbolic:
            embed = self.manifold.expmap0(embed)   # B x D

        return embed

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent variables
        onto the image space.

        Params:
            z (Tensor): Latent variable [B x D]

        Returns:
            result (Tensor) [B x C x H x W]
        """

        if self.hyperbolic:
            z = self.manifold.logmap0(z)    # B x D
        # result = self.decoder_input(z)  # B x (E1 * H' * W')
        # result = result.view(-1, self.hidden_dims[-1], self.hw_prime, self.hw_prime)     # B x E x H' x W'
        # result = self.decoder(result)   # B x E1 x H'' x W''
        # result = self.final_layer(result)   # B x C2 x H x W
        result = self.full_decoder(z)

        return result

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.

        Params:
            x (Tensor): input image Tensor [B x C x H x W]

        Returns:
            (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


#####################

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class VQ_CVAE(nn.Module):

    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )
        # self.encoder = nn.Sequential(
        #     Encoder(hidden_dims=[d, d, d, d], in_channels=1),
        #     nn.Linear(d * d_prime, latent_dims)
        # )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        # for l in self.modules():
        #     if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
        #         l.weight.detach().normal_(0, 0.02)
        #         torch.fmod(l.weight, 0.04)
        #         nn.init.constant_(l.bias, 0)

        # import pdb; pdb.set_trace()
        # self.encoder.encoder[-1].weight.detach().fill_(1 / 40)

        # self.emb.weight.detach().normal_(0, 0.02)
        # torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        import pdb; pdb.set_trace()
        z_e = self.encode(x)
        z_q, _ = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb
