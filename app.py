from pathlib import Path
import json

from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "gnn_esol.pt"
CONFIG_PATH = ROOT / "models" / "model_config.json"

app = Flask(__name__)
DEVICE = torch.device("cpu")

HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]


def atom_features(atom):
    hybrid = [float(atom.GetHybridization() == h) for h in HYBRIDIZATIONS]
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 5.0,
        atom.GetFormalCharge() / 3.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 250.0,
        *hybrid,
    ]


def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("The supplied SMILES string is not valid.")

    x = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float
    )

    edges = []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index), molecule


class MoleculeGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.regressor(x).view(-1)


def load_model():
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        return None, "Model files are missing. Run the notebook first to train and save the model."

    config = json.loads(CONFIG_PATH.read_text())
    model = MoleculeGNN(
        in_channels=config["in_channels"],
        hidden_channels=config.get("hidden_channels", 64),
    ).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, None


MODEL, MODEL_ERROR = load_model()


def predict(smiles):
    graph, molecule = smiles_to_graph(smiles)
    graph = graph.to(DEVICE)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        value = MODEL(graph.x, graph.edge_index, batch).item()

    details = {
        "atoms": molecule.GetNumAtoms(),
        "bonds": molecule.GetNumBonds(),
        "molecular_weight": round(Descriptors.MolWt(molecule), 2),
    }
    return value, details


@app.route("/", methods=["GET", "POST"])
def index():
    smiles = "CCO"
    prediction = None
    details = None
    error = MODEL_ERROR

    if request.method == "POST":
        smiles = request.form.get("smiles", "").strip()
        if not smiles:
            error = "Please enter a SMILES string."
        elif MODEL is None:
            error = MODEL_ERROR
        else:
            try:
                prediction, details = predict(smiles)
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        smiles=smiles,
        prediction=prediction,
        details=details,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
