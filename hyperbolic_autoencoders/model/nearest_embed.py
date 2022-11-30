import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
from hypmath import metrics


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    input - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    hyperbolic (bool): Bool for if embeddings are in hyperbolic space.
    bounded_measure (bool): Boolean to compute the distance using a bounded measure in Euclidean space. So, instead
        of computing Euclidean distance, computes the cosine similarity. In hyperbolic space, instead of computing
        hyperbolic distance, calculates the Minkowski inner product (which is not bounded, but is the hyperbolic
        analog of cosine similarity as per
        https://math.stackexchange.com/questions/2852458/bounded-similarity-measure-for-points-in-hyperbolic-space).
    """
    @staticmethod
    def forward(ctx, input, emb, hyperbolic, bounded_measure):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist_dim_euc = 1
        dist_dim_hyper = 1

        if hyperbolic:
            # Minkowski inner product
            if bounded_measure:
                raise NotImplementedError

            # Hyperbolic distance.
            else:
                dist = metrics.PoincareDistance(emb_expanded, x_expanded, dist_dim_hyper)
        else:

            # Cosine distance.
            if bounded_measure:
                dist_ = torch.norm(x_expanded / torch.norm(input) - emb_expanded / torch.norm(emb), 2, dist_dim_euc)
                dist = (dist_ ** 2) / 2

            # Euclidean distance.
            else:
                dist = torch.norm(x_expanded - emb_expanded, 2, dist_dim_euc)

        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        # Match vector norm of quantized embedding to that of un-quantized embedding if using bounded measure of
        # distance.
        if bounded_measure:
            if hyperbolic:
                raise NotImplementedError
            else:
                result = result / torch.norm(result) * torch.norm(emb)

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb, hyperbolic: bool, bounded_measure: bool):
    return NearestEmbedFunc().apply(x, emb, hyperbolic, bounded_measure)


class NearestEmbed(nn.Module):

    def __init__(self, num_embeddings, embeddings_dim, hyperbolic=True, bounded_measure: bool = False):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))
        self.hyperbolic = hyperbolic
        self.bounded_measure = bounded_measure

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight, self.hyperbolic, self.bounded_measure)

    def reinit_weights(self, scheme='data', dataloader=None):
        """Reinitialize the weights of the codebooks."""

        if scheme == 'data':
            assert dataloader, "For data-dependent initialization, you need to provide a dataloader."





# import numpy as np
# import torch
# from torch import nn
# from torch.autograd import Function, Variable
# from hypmath import metrics
#
#
# class NearestEmbedFunc(Function):
#     """
#     Input:
#     ------
#     x - (batch_size, emb_dim, *)
#         Last dimensions may be arbitrary
#     emb - (emb_dim, num_emb)
#     """
#
#     @staticmethod
#     def forward(ctx, input, emb, hyperbolic):
#         """
#         https://lars76.github.io/2020/07/24/implementing-poincare-embedding.html
#         Params:
#             ctx:
#             input:
#             emb:
#         :return:
#         """
#
#         if input.size(1) != emb.size(0):
#             raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
#                                format(input.size(1), emb.size(0)))
#
#         # save sizes for backward
#         ctx.batch_size = input.size(0)
#         ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
#         ctx.emb_dim = emb.size(0)
#         ctx.num_emb = emb.size(1)
#         ctx.input_type = type(input)
#         ctx.dims = list(range(len(input.size())))
#
#         # expand to be broadcast-able
#         x_expanded = input.unsqueeze(-1)
#         num_arbitrary_dims = len(ctx.dims) - 2
#         if num_arbitrary_dims:
#             emb_expanded = emb.view(
#                 emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
#         else:
#             emb_expanded = emb
#
#         # Find nearest neighbors in space.
#         dist_dim = 0
#         dist = metrics.PoincareDistance(
#             emb_expanded, x_expanded, dist_dim) if hyperbolic else torch.norm(x_expanded - emb_expanded, 2, dist_dim)
#
#         _, argmin = dist.min(-1)
#         shifted_shape = [input.shape[0], *
#                          list(input.shape[2:]), input.shape[1]]
#         result = emb.t().index_select(0, argmin.view(-1)
#                                       ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])
#
#         ctx.save_for_backward(argmin)
#         return result.contiguous(), argmin
