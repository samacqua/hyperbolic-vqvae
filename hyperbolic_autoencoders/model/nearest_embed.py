import numpy as np
import torch
from torch import nn
from torch.autograd import Function
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
    def forward(ctx, input: torch.Tensor, emb: torch.Tensor, hyperbolic: bool, bounded_measure: bool):
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

        k = emb.shape[1]
        batch_size, d, h, w = input.shape

        # find nearest neighbors
        dist_dim_euc = 1
        dist_dim_hyper = 1

        if hyperbolic:
            # Minkowski inner product
            if bounded_measure:
                raise NotImplementedError

            # Hyperbolic distance.
            else:

                # def ps(t1, t2):
                #     assert t1.numel() == t2.numel()
                #     return torch.isclose(t1, t2).sum() / t1.numel()
                #
                # print("quantize 1", end='\t')
                # t0 = time.time()
                # dists0 = metrics.PoincareDistance(emb_expanded, x_expanded, dist_dim_hyper)
                # _, argmin0 = dists0.min(-1)
                # shifted_shape = [input.shape[0], * list(input.shape[2:]), input.shape[1]]
                # result0 = emb.t().index_select(0, argmin0.view(-1)
                #                               ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])
                #
                # t1 = time.time()
                # print(t1 - t0)
                # batch_size, d, h, w = input.shape
                #
                # result1 = torch.zeros_like(input)
                # distances1 = torch.zeros(batch_size, h, w, emb.shape[1])
                # argmin1 = torch.zeros(batch_size, h, w)
                #
                #
                # print("quantize 2", end='\t')
                # for batch_i in range(batch_size):
                #     for i in range(h):
                #         for j in range(w):
                #             patch = input[batch_i, :, i, j]
                #
                #             dists = metrics.PoincareDistance(patch.unsqueeze(-1), emb, 0)
                #             distances1[batch_i, i, j, :] = dists
                #
                #             # slowest but most obviously correct way.
                #             # k = emb.shape[1]
                #             # dists = np.zeros(k)
                #             # for cbi in range(k):
                #             #     cb = emb[:, cbi]
                #             #     dist_ = metrics.PoincareDistance(patch, cb, 0)
                #             #     dists[cbi] = dist_
                #
                #             patch_argmin = dists.argmin()
                #             patch_quantized = emb[:,patch_argmin]
                #
                #             result1[batch_i, :, i, j] = patch_quantized
                #             argmin1[batch_i, i, j] = patch_argmin
                #
                #         a = input[batch_i, :, i, :]      # [7, 8], emb=[7, 15]
                #         d = distances1[batch_i, i, :, :]  # [8, 15]
                #         f = metrics.PoincareDistance
                #         su = f(a.unsqueeze(-1), emb.view(emb.shape[0], 1, emb.shape[1]), 0)
                #         try:
                #             assert ps(su, d) == 1
                #         except AssertionError:
                #             import pdb; pdb.set_trace()
                #
                # t2 = time.time()
                # print(t2 - t1)
                # print("quantize 3", end='\t')
                dist = torch.zeros(batch_size, h, w, emb.shape[1])
                for batch_i in range(batch_size):
                    dist[batch_i, :, :, :] = metrics.PoincareDistance(
                            input[batch_i, :, :, :].unsqueeze(-1),
                            emb.view(emb.shape[0], 1, 1, emb.shape[1]), 0)

                argmin = dist.argmin(-1)

                shifted_shape = [input.shape[0], *list(input.shape[2:]), input.shape[1]]
                result = emb.t().index_select(0, argmin.view(-1)
                                               ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

                # t3 = time.time()
                # assert ps(dist, distances1) == 1
                # assert ps(result, result1) == 1
                # print(t3 - t2)
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


def nearest_embed(x, emb, hyperbolic: bool, bounded_measure: bool, n_groups: int):

    batch_size, d, h, w = x.shape
    group_size = d // n_groups

    split_x = torch.split(x, group_size, dim=1)
    result = torch.zeros_like(x)
    argmins = torch.zeros(batch_size, n_groups, h, w).type(torch.LongTensor)
    for i, subtensor in enumerate(split_x):
        result[:, i*group_size:(i+1)*group_size, :, :], argmins[:,i,:,:] = NearestEmbedFunc().apply(subtensor, emb, hyperbolic, bounded_measure)

    result /= n_groups ** 0.5
    # out = NearestEmbedFunc().apply(x, emb, hyperbolic, bounded_measure)
    return result, argmins


class NearestEmbed(nn.Module):

    def __init__(self, num_embeddings, embeddings_dim, hyperbolic=True, bounded_measure: bool = False, n_groups: int = 1):
        super(NearestEmbed, self).__init__()

        assert embeddings_dim % n_groups == 0
        subtensor_size = embeddings_dim // n_groups

        self.weight = nn.Parameter(torch.rand(subtensor_size, num_embeddings))

        self.hyperbolic = hyperbolic
        self.bounded_measure = bounded_measure
        self.n_groups = n_groups

        self.k = num_embeddings
        self.d = subtensor_size

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight, self.hyperbolic,
                             self.bounded_measure, self.n_groups)

    def reinit_weights(self, scheme='data', encodings=None):
        """Reinitialize the weights of the codebooks."""

        if scheme == 'data':

            assert encodings is not None, "For data-dependent initialization, you need to provide encodings of the data."
            assert encodings.shape[1] > self.k, "Must encode >= the number of codebook vectors."
            assert encodings.shape[0] == self.d * self.n_groups, "Provided encodings have the incorrect dimensions."

            # Split encodings into subtensors.
            group_size = self.d
            split_encodings = torch.split(encodings, group_size, dim=0)
            flattened_encodings = torch.concat(split_encodings, dim=1)

            # Randomly select k encodings.
            perm = torch.randperm(flattened_encodings.shape[1])
            idx = perm[:self.k]
            samples = flattened_encodings[:,idx]

            assert samples.shape == (self.d, self.k)

            self.weight = nn.Parameter(samples.detach())
        else:
            raise ValueError("Unrecognized initialization scheme " + str(scheme))





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

