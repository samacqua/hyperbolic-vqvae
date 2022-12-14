from data_loader import binary_tree
import matplotlib.pyplot as plt
import torch
import utils.plot_utils as plot_utils
import os
import torch.functional as F


def h2p(x, manifold):
    return manifold.logmap0(x)
    # x0, xrest = F.split_axis(x, indices_or_sections=(1,), axis=-1)
    # ret = (xrest / F.broadcast_to(1 + x0, xrest.shape))
    # return ret


def visualize_embeddings(data_loader, model, hyperbolic: bool = False, save_dir='./'):
    """Visualize the embeddings."""

    N = len(data_loader.data)
    # Get encoding of actual tree.
    z, _ = model.encode(torch.tensor(data_loader.data, dtype=torch.float64))
    z = z.detach()

    # Get encoding of perturbations of tree.
    rand_data = []
    for _ in range(N):
        sampled, *_ = next(iter(data_loader))
        rand_data += list(sampled)

        if len(rand_data) > N:
            break

    rand_data = torch.stack(rand_data[:N])
    z_prob, _ = model.encode(rand_data)
    z_prob = z_prob.detach()

    # ???
    if hyperbolic:
        z_vis = h2p(z, model.manifold)
        z_prob_vis = h2p(z_prob, model.manifold)
    else:
        z_vis = z
        z_prob_vis = z_prob

    # Set up figure.
    plot_utils.getfig((4, 4))
    plt.box('off')
    plt.xticks([])
    plt.yticks([])

    # Plot the connections of the embeddings of the actual tree.
    for i in range(len(data_loader.data)):
        for j in range(i):
            if binary_tree.hamming_distance(
                    data_loader.data[i], data_loader.data[j]) == 1:
                plt.plot(
                    [z_vis[i, 0], z_vis[j, 0]], [z_vis[i, 1], z_vis[j, 1]],
                    lw=1, color='gray', zorder=1)

    # Plot the embeddings of the nodes of the tree.
    plt.scatter(
        z_vis[:, 0], z_vis[:, 1], s=(300 / data_loader.data.sum(axis=-1) ** 2),
        c='#F15A29', zorder=10)

    # Plot the center of the embedding space.
    plt.scatter(0, 0, s=100, c='magenta', zorder=20, marker='x')

    # Plot the embeddings of the noised parts of the tree.
    plt.scatter(
        z_prob_vis[:, 0], z_prob_vis[:, 1],
        s=10, c='#4B489E', zorder=5)

    plt.show()

    plt.savefig(os.path.join(save_dir, 'embedding.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'embedding.png'), bbox_inches='tight', dpi=400)
