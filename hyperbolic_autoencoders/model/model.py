import torch
from base import BaseModel
from torch import nn
from torch.nn import functional as F
from hypmath import WrappedNormal, poincareball, metrics
from torch.distributions import Normal
from typing import TypeVar, List, Optional, Tuple, Union
from .nearest_embed import NearestEmbed, nearest_embed
import logging
import numpy as np
from base import BaseDataLoader


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


class FlatEncoder(BaseModel):
    """Flat encoder. Sequence of linear layers + non-linearity."""

    def __init__(self, in_size: int, hidden_dims: List[int], nonlinearity=nn.Tanh):
        """Initializes the encoder.

        Params:
            hidden_dims (List[int]): List of the depth of each convolutional layer.
            in_channels: (int): The number of channels in the input image.
        """
        super(FlatEncoder, self).__init__()

        modules = []
        hidden_dims = [in_size] + hidden_dims

        for h_dim_in, h_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.Linear(h_dim_in, h_dim_out),
                    nonlinearity())
            )

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


class FlatDecoder(BaseModel):
    """Flat decoder. Sequence of linear layer + non-linearity."""

    def __init__(self, hidden_dims: List[int], out_size: int, nonlinearity=nn.Tanh):
        """Initializes the decoder.

        Params:
            hidden_dims (List[int]): List of the depth of each convolutional layer.
        """

        super(FlatDecoder, self).__init__()

        modules = []
        hidden_dims.append(out_size)

        for h_dim_in, h_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.Linear(h_dim_in, h_dim_out),
                    nonlinearity())
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


class FlatVAE(VanillaVAE):
    def __init__(self, data_shape: int = 255, hidden_d: int = 32, latent_d: int = 16, hyperbolic: bool = False):
        super(VanillaVAE, self).__init__()

        self.manifold = poincareball.PoincareBall(latent_d) if hyperbolic else None
        self.hyperbolic = hyperbolic

        self.encoder = nn.Sequential(
            nn.Linear(data_shape, hidden_d),
            nn.Tanh()
        )

        self.fc_mu = nn.Linear(hidden_d, latent_d)
        self.fc_var = nn.Linear(hidden_d, latent_d)

        self.decoder = nn.Sequential(
            nn.Linear(latent_d, hidden_d),
            nn.Tanh(),
            nn.Linear(hidden_d, data_shape)
        )

    def encode(self, x):
        emb = self.encoder(x)
        mu, log_var = self.fc_mu(emb), self.fc_var(emb)
        if self.hyperbolic:
            mu = self.manifold.expmap0(mu)   # B x D
            log_var = F.softplus(log_var)     # B x D
        return mu, log_var

    def decode(self, z):
        if self.hyperbolic:
            z = self.manifold.logmap0(z)    # B x D
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)  # (B x D, B x D)
        z = self.reparameterize(mu, log_var)  # B x D
        output = self.decode(z)  # B x C x H x W

        return output, mu, log_var

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


#####################


class ResBlock(BaseModel):
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


def make_resnet_encoder(num_channels, d, bn=True, final_d=None):
    final_d = final_d or d
    return nn.Sequential(
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

        nn.Conv2d(d, final_d, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(final_d),
    )


class VQVAE(BaseModel):

    def __init__(self, hidden_dims: Union[List[int], int] = 64, k: int = 10, bn: bool = True, commit_coef: float = 0.5,
                 num_channels: int = 3, hyperbolic: bool = False, custom_init: bool = False,
                 n_classes: Optional[int] = None, n_groups: int = 1, resnet: bool = False, **kwargs):
        super(VQVAE, self).__init__()

        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        d = hidden_dims[-1]
        d0 = hidden_dims[0]
        if d % n_groups != 0:
            raise ValueError("For Grouped vector-quantization, the embedding dimension must be divisible by the number of groups.")

        # Make encoder.
        if resnet and len(hidden_dims) > 2:
            raise ValueError("Resnet requires constant latent dimension or just 2 (normal + final). Given list.")

        self.encoder = make_resnet_encoder(num_channels, d0, bn, d) if resnet else Encoder(hidden_dims=hidden_dims,
                                                                                       in_channels=num_channels)
        hidden_dims = list(reversed(hidden_dims))

        if resnet:
            decoder_layers = [
                nn.ConvTranspose2d(d, d0, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(d0),
                nn.ReLU(inplace=False),

                ResBlock(d0, d0),
                nn.BatchNorm2d(d0),
                ResBlock(d0, d0),
            ]
            if n_classes:
                decoder_layers += [
                    nn.BatchNorm2d(d0),
                    nn.Flatten(),
                    nn.ReLU(inplace=False),
                    nn.LazyLinear(n_classes)
                ]
            else:
                decoder_layers += [
                    nn.ConvTranspose2d(d0, d0, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(d0),
                    nn.ReLU(inplace=False),
                    nn.ConvTranspose2d(
                        d0, num_channels, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()
                ]
        else:
            self.hidden_dims = hidden_dims.copy()
            decoder_layers = [
                Decoder(hidden_dims=hidden_dims),
                nn.Tanh()
            ]
            if n_classes:
                decoder_layers += [
                    nn.Flatten(),
                    nn.LazyLinear(n_classes)
                ]
            else:
                decoder_layers += [
                    nn.ConvTranspose2d(d, d0, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), output_padding=(1, 1)),
                    nn.BatchNorm2d(d0),
                    nn.LeakyReLU(),
                    nn.Conv2d(d0, out_channels=num_channels,
                              kernel_size=(3, 3), padding=1),
                    nn.Tanh()
                ]

        self.decoder = nn.Sequential(*decoder_layers)

        self.hyperbolic = hyperbolic
        self.manifold = poincareball.PoincareBall(d)

        self.k = k
        self.d = d
        self.emb = NearestEmbed(k, d, hyperbolic=hyperbolic, n_groups=n_groups, manifold=self.manifold)
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        if custom_init:
            for l in self.modules():
                if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):

                    l.weight.detach().normal_(0, 0.02)
                    torch.fmod(l.weight, 0.04)
                    nn.init.constant_(l.bias, 0)

            self.encoder[-1].weight.detach().fill_(1 / 40)

            self.emb.weight.detach().normal_(0, 0.02)
            torch.fmod(self.emb.weight, 0.04)

    def data_dependent_init(self, data_loader):
        """Initializes the codebooks based on the input data."""
        data_loader = BaseDataLoader(data_loader.dataset, batch_size=self.k, shuffle=True,
                                    validation_split=0, num_workers=data_loader.num_workers)

        data, target = next(iter(data_loader))
        encodings = self.encode(data)
        batch_size, d, dim1, dim2 = encodings.shape
        encodings = encodings.reshape(d, batch_size * dim1 * dim2)

        self.emb.reinit_weights('data', encodings=encodings)

    def encode(self, x):
        encoded = self.encoder(x)
        if self.hyperbolic:
            encoded = self.manifold.expmap0(encoded)  # B x D
        return encoded

    def decode(self, x):
        if self.hyperbolic:
            x = self.manifold.logmap0(x)    # B x D
        return self.decoder(x)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin, self.decode(z_e.detach()), self.decode(z_q.detach())

    def plot_codebooks(self):
        """Visualize the codebook_vectors."""
        reshaped_codebooks = self.emb.weight.detach().reshape(self.k, self.d, 1, 1)
        decoded_codebooks = self.decode(reshaped_codebooks)
        return decoded_codebooks.detach()

    def visualize_im_codebooks(self, x):
        """Visualize the codebooks that make up the inference on an example."""

        assert len(x.shape) == 3, "can only visualize 1 image at a time."

        # Get inference.
        x = x.unsqueeze(0)
        z_e = self.encode(x.detach())
        z_q, argmin = self.emb(z_e.detach(), weight_sg=True)
        recon = self.decode(z_q.detach()).detach().squeeze()

        # Reshape vals.
        argmin = argmin.squeeze()
        bottleneck_h, bottleneck_w = argmin.shape
        x = x.squeeze()
        data_h, data_w = x.shape
        step_n = data_h // bottleneck_h

        import matplotlib.pyplot as plt
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 8))

        # Plot original image.
        ax0.imshow(x, cmap='gray', vmin=0, vmax=1)

        ax0.set_xticks(np.arange(-0.5, data_w, step=step_n))
        ax0.set_yticks(np.arange(-0.5, data_h, step=step_n))
        ax0.set_xticklabels(np.arange((data_w//step_n)+1, step=1))
        ax0.set_yticklabels(reversed(np.arange((data_h//step_n)+1, step=1)))
        ax0.grid()

        # Plot reconstruction.
        ax1.imshow(recon, cmap='gray', vmin=0, vmax=1)

        ax1.set_xticks(np.arange(-0.5, data_w, step=step_n))
        ax1.set_yticks(np.arange(-0.5, data_h, step=step_n))
        ax1.set_xticklabels(np.arange((data_w//step_n)+1, step=1))
        ax1.set_yticklabels(reversed(np.arange((data_h//step_n)+1, step=1)))
        ax1.grid()

        # Plot argmins.
        for i in range(bottleneck_h):
            for j in range(bottleneck_w):
                c = argmin[i][j].item()
                ax2.text(i + 0.5, j + 0.5, str(c), va='center', ha='center')

        ax2.set_xlim(0, bottleneck_w)
        ax2.set_ylim(0, bottleneck_h)
        ax2.set_xticks(np.arange(bottleneck_w+1))
        ax2.set_yticks(np.arange(bottleneck_h+1))
        ax2.grid()

        # Plot codebooks.
        codebooks = self.plot_codebooks().squeeze()
        k, cbh, cbw = codebooks.shape
        cbw *= int(np.sqrt(k))
        cbh *= int(np.sqrt(k))
        codebooks = codebooks.reshape(cbh, cbw)
        ax3.imshow(codebooks, cmap='gray', vmin=0, vmax=1)

        ax3.set_xticks(np.arange(-0.5, cbw, step=step_n))
        ax3.set_yticks(np.arange(-0.5, cbh, step=step_n))
        ax3.set_xticklabels(np.arange((cbw // step_n) + 1, step=1))
        ax3.set_yticklabels(reversed(np.arange((cbh // step_n) + 1, step=1)))
        ax3.grid()

        plt.tight_layout()
        plt.show()
