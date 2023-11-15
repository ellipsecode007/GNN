import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
num_nodes = max(max(u, v) for u, v in edges) + 1
graph = DGLGraph(edges)
node_features = torch.randn(num_nodes, 16)

train_edges, val_edges = train_test_split(edges, test_size=0.2, random_state=42)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, out_feats, 'mean')

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

class LinkPredictionModel(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(LinkPredictionModel, self).__init__()
        self.sage = GraphSAGE(in_feats, hidden_size, out_feats)
        self.prediction = nn.Linear(out_feats * 2, 1)

    def forward(self, g, node_features, edge_src, edge_dst):
        node_embedding = self.sage(g, node_features)
        src_embed = node_embedding[edge_src]
        dst_embed = node_embedding[edge_dst]
        pred = torch.cat([src_embed, dst_embed], dim=1)
        pred = self.prediction(pred)
        return pred.squeeze()

train_src, train_dst = zip(*train_edges)
val_src, val_dst = zip(*val_edges)

model = LinkPredictionModel(node_features.shape[1], 16, 8)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(graph, node_features, torch.tensor(train_src), torch.tensor(train_dst))
    train_targets = torch.randn(len(train_edges))

    loss = nn.MSELoss()(predictions, train_targets.float())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

with torch.no_grad():
    val_predictions = model(graph, node_features, torch.tensor(val_src), torch.tensor(val_dst))
    val_targets = torch.randn(len(val_edges))
    val_mae = mean_absolute_error(val_targets, val_predictions.numpy())
    print(f"Validation MAE: {val_mae}")

test_edges = [(0, 2), (1, 3), (4, 2)]
num_test_nodes = max(max(u, v) for u, v in test_edges) + 1
test_graph = DGLGraph(test_edges)
test_node_features = torch.randn(num_test_nodes, 16)

test_src, test_dst = zip(*test_edges)

with torch.no_grad():
    test_predictions = model(test_graph, test_node_features, torch.tensor(test_src), torch.tensor(test_dst))
    print("Test Predictions:")
    print(test_predictions.numpy())
