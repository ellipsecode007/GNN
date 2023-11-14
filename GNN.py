import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import mean_absolute_error

ratings_df = pd.read_csv("ratings.csv")

user_encoder = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
movie_encoder = {movie_id: idx for idx, movie_id in enumerate(ratings_df['movieId'].unique())}

ratings_df['userId'] = ratings_df['userId'].map(user_encoder)
ratings_df['movieId'] = ratings_df['movieId'].map(movie_encoder)

edge_index = torch.tensor([ratings_df['userId'], ratings_df['movieId']], dtype=torch.long)

x = torch.tensor(ratings_df['rating'].values, dtype=torch.float).view(-1, 1)

data = Data(x=x, edge_index=edge_index)

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)
#         self.final_layer = nn.Linear(out_channels, 1) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
#         x = self.final_layer(x).squeeze() 
        return x

model = GraphSAGEModel(in_channels=1, out_channels=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

loader = DataLoader([data], batch_size=1, shuffle=True)

model.train()
for epoch in range(100):
    total_loss = 0.0
    total_mae = 0.0
    
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.x)
        loss.backward()
        optimizer.step()
        
        mae = mean_absolute_error(batch.x.detach().numpy(), output.detach().numpy())
        total_mae += mae.item()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    
    print(f'Epoch {epoch + 1}/{100}, Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}')
