import geoopt
import torch

ball = geoopt.PoincareBall(1)


def PoincareDistance(X1, X2, dim=-1):
    """Calculates the poincare distance between two points on the poincare ball,
        used for calculation of the Silhouette score.
    """
    return ball.dist(torch.Tensor(X1), torch.Tensor(X2), dim=dim)

if __name__ == '__main__':

    p1 = torch.tensor([0.5, 0.5])
    p2 = torch.tensor([-0.5, -0.5])

    print(PoincareDistance(p1, p2))
