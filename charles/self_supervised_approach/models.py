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
            loss: loss value.
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)

        # computes similarity matrix
        xcs = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)

        # fill diagonal with -inf for same elements
        xcs[torch.eye(xcs.size(0)).bool()] = float("-inf")

        # Positive pair indices
        target = torch.arange(batch_size, device=xcs.device)
        target = torch.cat([target + batch_size, target])

        xcs /= self.temperature

        loss = F.cross_entropy(xcs, target, reduction="mean")
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


# -------------------------------------------------------------------
# BYOL Implementation
# -------------------------------------------------------------------

class BYOL(nn.Module):
    """
    BYOL model that includes:
      - Online network (Encoder + Projector + Predictor)
      - Target network (Encoder + Projector)
    """
    def __init__(self, base_encoder, projection_dim=128, hidden_dim=2048, moving_avg_decay=0.99, device='cuda'):
        super(BYOL, self).__init__()
        self.device = device
        self.moving_avg_decay = moving_avg_decay

        # -------------------------
        # Online network
        # -------------------------
        self.online_encoder = base_encoder
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 30, 75).to(device)
            encoder_output = self.online_encoder(dummy_input)
            encoder_output_dim = encoder_output.shape[-1]
            if len(encoder_output.shape) > 1:
                encoder_output_dim = encoder_output.shape[1]

        self.online_projector = ProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # -------------------------
        # Target network
        # -------------------------
        self.target_encoder = base_encoder
        self.target_projector = ProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

        # Make sure target always starts as a copy of online
        self._init_target_network()

    def _init_target_network(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        for param_o, param_t in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        """
        Momentum update of target network.
        theta_t = m * theta_t + (1 - m) * theta_o
        """
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data = param_t.data * self.moving_avg_decay + \
                           param_o.data * (1.0 - self.moving_avg_decay)

        for param_o, param_t in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_t.data = param_t.data * self.moving_avg_decay + \
                           param_o.data * (1.0 - self.moving_avg_decay)

    def forward(self, x1, x2):
        """
        x1, x2: Two augmented views of the same batch.
        Returns a tuple: p1, z1, p2, z2
         - p1, p2: the online prediction
         - z1, z2: the target projection
        """
        # ------------------------
        # online side
        # ------------------------
        o1 = self.online_encoder(x1)                     # batch x feat
        o1 = self.online_projector(o1)                   # batch x proj_dim
        p1 = self.online_predictor(o1)                   # batch x proj_dim
        p1 = F.normalize(p1, dim=1)

        o2 = self.online_encoder(x2)
        o2 = self.online_projector(o2)
        p2 = self.online_predictor(o2)
        p2 = F.normalize(p2, dim=1)

        # ------------------------
        # target side
        # ------------------------
        with torch.no_grad():
            t1 = self.target_encoder(x1)
            t1 = self.target_projector(t1)
            t1 = F.normalize(t1, dim=1)

            t2 = self.target_encoder(x2)
            t2 = self.target_projector(t2)
            t2 = F.normalize(t2, dim=1)

        return p1, t1, p2, t2

    def compute_loss(self, x1, x2):
        """
        Compute BYOL loss given two batches of augmented data.
        The standard BYOL loss is the mean squared error of normalized predictions and targets:
            L = 2 - 2 * cos_sim(p1, t2) + 2 - 2 * cos_sim(p2, t1)
        """
        p1, t1, p2, t2 = self.forward(x1, x2)

        # BYOL uses negative cosine similarity or MSE on normalized vectors.
        # simple version: L = 2 - 2 * cos_sim(p1, t2) ...
        loss_fn = lambda p, t: 2 - 2 * F.cosine_similarity(p, t, dim=-1)

        loss_1 = loss_fn(p1, t2)
        loss_2 = loss_fn(p2, t1)
        loss = (loss_1 + loss_2).mean()
        return loss

# -----------------------
# For fine tuning network for event classification
# ------------------------
class FrozenEncoderBinaryClassifier(nn.Module):
    def __init__(self, encoder, embedding_dim=512, num_classes=1):
        super(FrozenEncoderBinaryClassifier, self).__init__()
        self.encoder = encoder
        # freeze encoder parameters (SIMCLR part)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # classification head: maps embedding_dim to num_classes (binary classification 0 or 1 success)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # pass input through encoder with frozen wights
        with torch.no_grad():
            emb = self.encoder(x)
        # L2 normalize embeddings
        emb = F.normalize(emb, p=2, dim=1)

        logits = self.classifier(emb)
        return logits
