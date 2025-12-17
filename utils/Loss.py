
import nn

class BNPELoss(nn.Module):
        def __init__(self, estimator, prior, lmbda=100.0):
            super().__init__()
            self.estimator = estimator
            self.prior = prior
            self.lmbda = lmbda
        def forward(self, theta, x):
            theta_prime = torch.roll(theta, 1, dims=0)
            log_p, log_p_prime = self.estimator(
                torch.stack((theta, theta_prime)),
                x,
            )
            l0 = -log_p.mean()
            lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square()
            return l0 + self.lmbda * lb