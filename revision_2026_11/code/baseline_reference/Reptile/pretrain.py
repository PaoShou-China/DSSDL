import torch
from Reptile import Reptile
from datasets import dataset_dict


# The Reptile class is expected to manage its own model instance (self.network)
reptile = Reptile(dataset_dict)

# Train the model
reptile.train(n_epochs=200)

torch.save(reptile.network.state_dict(), "pretrain.pt")
