import torch
from torch import nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, support: dict, query: dict):
        """
        ProtoNet forward-pass.

        Args:
            support (dict): A dictionary containing the support set. 
                The support set dict must contain the following keys:
                    - audio: A tensor of shape (n_support, n_channels, n_samples)
                    - label: A tensor of shape (n_support) with label indices
                    - classes: A tensor of shape (n_classes) containing the list of classes in the current episode
            query (dict): A dictionary containing the query set.
                The query set dict must contain the following keys:
                    - audio: A tensor of shape (n_query, n_channels, n_samples)
        
        Returns:
            predictions (torch.Tensor): A tensor of shape (n_query) containing the argmax predictions
        """

        support["embeddings"] = self.backbone(support["audio"])
        query["embeddings"] = self.backbone(query["audio"])

        # Group support embeddings by class
        support_embeddings = []
        for idx in range(len(support["classes"])):
            embeddings = support["embeddings"][support["labels"] == idx]
            support_embeddings.append(embeddings)
        support_embeddings = torch.stack(support_embeddings)

        # Compute the prototypes for each class
        prototypes = support_embeddings.mean(dim=1)
        support["prototypes"] = prototypes

        # Compute the distances between each query and prototype
        distances = torch.cdist(
            query["embeddings"].unsqueeze(0), 
            prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        # Square the distances to get the sq euclidean distance
        distances = distances ** 2
        logits = -distances
        
        predicitions = torch.argmax(logits, dim=1)
        return predicitions

