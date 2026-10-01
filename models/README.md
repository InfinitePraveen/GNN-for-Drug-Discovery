# Models

After the notebook finishes training, it writes:

- `gnn_esol.pt` — PyTorch model weights.
- `model_config.json` — model architecture and feature configuration.

These generated files are ignored by Git so the repository stays lightweight. The Flask app loads them locally.
