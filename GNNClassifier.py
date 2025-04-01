import torch
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# Example graphs: 3 graphs, each with different structures and features
edge_index_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Graph 1 edges
x_1 = torch.tensor([[1, 2], [2, 3]], dtype=torch.float)  # Graph 1 node features
y_graph_1 = torch.tensor([1], dtype=torch.long)  # Graph 1 label (e.g., human vs AI)

edge_index_2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # Graph 2 edges
x_2 = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float)  # Graph 2 node features
y_graph_2 = torch.tensor([0], dtype=torch.long)  # Graph 2 label (e.g., human vs AI)

# More example graphs...
edge_index_3 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Graph 3 edges
x_3 = torch.tensor([[3, 4], [4, 5]], dtype=torch.float)  # Graph 3 node features
y_graph_3 = torch.tensor([1], dtype=torch.long)  # Graph 3 label (e.g., human vs AI)

# Create Data objects for each graph
data1 = Data(x=x_1, edge_index=edge_index_1, y=y_graph_1)
data2 = Data(x=x_2, edge_index=edge_index_2, y=y_graph_2)
data3 = Data(x=x_3, edge_index=edge_index_3, y=y_graph_3)

# Stack the graphs into a DataLoader for batching
dataset = [data1, data2, data3]
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define a GCN model for graph classification
class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNGraphClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc1 = torch.nn.Linear(32, 64)  # Fully connected layer for graph-level representation
        self.fc2 = torch.nn.Linear(64, out_channels)  # Output layer for classification
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Graph-level pooling (mean pooling over nodes)
        x = global_mean_pool(x, batch)
        
        # Fully connected layers for classification
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x

# Initialize and train the model
model = GCNGraphClassifier(in_channels=2, out_channels=2)  # Example: 2 classes for graph classification (human vs AI)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop for graph classification (with multiple graphs)
model.train()
for epoch in range(100):
    total_loss = 0
    for batch_data in loader:
        optimizer.zero_grad()
        out = model(batch_data)
        loss = criterion(out, batch_data.y)  # Use graph-level labels (batch_data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

# Inference
model.eval()
with torch.no_grad():
    for batch_data in loader:
        out = model(batch_data)
        _, predicted = out.max(dim=1)
        print(f'Predicted graph class: {predicted}')
