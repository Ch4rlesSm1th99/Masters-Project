import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    """
    Base CNN encoder for extracting features from the input data.
    """
    def __init__(self, input_channels=1):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)

        return x


class ProjectionHead(nn.Module):
    """
    Projection head as per SimCLR paper: a 2-layer MLP with ReLU activation.
    """
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimCLR(nn.Module):
    """
    SimCLR model that includes an encoder and a projection head.
    """
    def __init__(self, base_encoder, projection_dim=128, temperature=0.5, device='cuda'):
        """
        Args:
            base_encoder (nn.Module): The encoder network to extract features.
            projection_dim (int): Output dimension of the projection head.
            temperature (float): Temperature parameter for the NT-Xent loss.
            device (str): Device to run the computations on.
        """
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        self.device = device

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 30, 75).to(device)
            encoder_output = self.encoder(dummy_input)
            encoder_output_dim = encoder_output.shape[-1]
            if len(encoder_output.shape) > 1:
                encoder_output_dim = encoder_output.shape[1]

        self.projection_head = ProjectionHead(input_dim=encoder_output_dim, output_dim=projection_dim)

        self.temperature = temperature

    def forward(self, x):
        """
        Forward pass through encoder and projection head.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Projected features.
        """
        h = self.encoder(x)
        z = self.projection_head(h)
        z = F.normalize(z, dim=1)
        return z

    def compute_loss(self, z_i, z_j):
        """
        Compute the NT-Xent loss.

        Args:
            z_i (torch.Tensor): Projected features from the first set of augmentations.
            z_j (torch.Tensor): Projected features from the second set of augmentations.

        Returns:
            torch.Tensor: Loss value.
        """
        return self.nt_xent_loss(z_i, z_j)

    def nt_xent_loss(self, z_i, z_j):
        """
        NT-Xent loss function as used in SimCLR.

        Args:
            z_i (torch.Tensor): Projected features from the first set of augmentations.
            z_j (torch.Tensor): Projected features from the second set of augmentations.

        Returns:
            torch.Tensor: Loss value.
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2 * batch_size, dim]

        # compute similarity matrix
        sim = torch.matmul(z, z.T)  # [2 * batch_size, 2 * batch_size]
        sim = sim / self.temperature

        # mask to remove self-comparisons
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        sim = sim.masked_fill(mask, float('-inf'))

        # positive pairs
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.masked_fill(mask, 0)
        labels = labels / labels.sum(dim=1, keepdim=True)

        loss = - (labels * F.log_softmax(sim, dim=1)).sum(dim=1).mean()
        return loss


def SimCLR_topk_accuracy(z_i, z_j, temperature=0.5, top_k=1):
    """
    Computes Top-k accuracy for self-supervised embeddings.

    Args:
        z_i (torch.Tensor): Projected features from the first set of augmentations. Shape: [batch_size, dim]
        z_j (torch.Tensor): Projected features from the second set of augmentations. Shape: [batch_size, dim]
        temperature (float): Temperature parameter.
        top_k (int): The 'k' in Top-k accuracy.

    Returns:
        float: Top-k accuracy as a percentage.
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    # compute similarity matrix
    sim = torch.matmul(z, z.T) / temperature

    # mask self-similarity from matrix
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, float('-inf'))

    # get the indices of the top k most similar embeddings
    _, indices = sim.topk(k=top_k, dim=1)

    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)  # positive pair indices

    # expand labels to match the size of indices
    labels = labels.unsqueeze(1).expand(-1, top_k)

    # Compare indices with labels
    correct = (indices == labels).any(dim=1).float()

    # score
    accuracy = correct.mean().item() * 100

    return accuracy
