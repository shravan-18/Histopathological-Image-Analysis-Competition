import cv2
import numpy as np
import os
     
from scipy.spatial.distance import cdist
from skimage.measure import label
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage import exposure


import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch

import networkx as nx


class HistopathologyGraphDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        for label in ['0', '1']:  # Assuming binary classification
            label_dir = os.path.join(root_dir, label)
            self.image_paths.extend([(label, os.path.join(label_dir, fname)) for fname in os.listdir(label_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label, image_path = self.image_paths[idx]
        labeled_image = self.preprocess_and_segment(image_path)
        features = self.extract_features(labeled_image)
        coords = np.array([prop.centroid for prop in regionprops(labeled_image)])
        G = self.construct_graph(features, coords)
        
        if G is None or len(coords) == 0:
            return None  # Skip this sample
        
        pyg_data = self.graph_to_pyg_data(G, label=int(label))
        return pyg_data

    @staticmethod
    def preprocess_and_segment(image_path):
        image = cv2.imread(image_path)
        gray_image = rgb2gray(image)
        gray_image_8bit = (gray_image * 255).astype(np.uint8)
        _, thresh_image = cv2.threshold(gray_image_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        labeled_image = label(thresh_image)
        return labeled_image

    @staticmethod
    def extract_features(labeled_image):
        props = regionprops(labeled_image)
        features = [[prop.area, prop.perimeter, prop.eccentricity] for prop in props]
        return np.array(features)

    @staticmethod
    def construct_graph(features, coords, distance_threshold=30):
        if len(coords) == 0:
            return None
        
#         print("Objects detected in the image")
        G = nx.Graph()
        for i, feature in enumerate(features):
            G.add_node(i, feature=np.array(feature))

        if len(coords) > 1:  # Ensure there are at least two points to form an edge
            dist_matrix = cdist(coords, coords, 'euclidean')
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    if dist_matrix[i, j] < distance_threshold:
                        G.add_edge(i, j)
        return G

    @staticmethod
    def graph_to_pyg_data(G, label):
        node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])
        edge_indices = np.array(list(G.edges())).T
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

def collate_geometric(samples):
    # Filter out None values in case of skipped samples
    samples = list(filter(None, samples))
    if not samples:
        return None  # Return None if all samples were filtered out
    batch = Batch.from_data_list(samples)
    return batch

root_dir = 'Root'
dataset = HistopathologyGraphDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, collate_fn=collate_geometric)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features=3, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels*4)
        self.conv2 = GCNConv(hidden_channels*4, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)  # Pooling
        x = self.classifier(x)
        return nn.LogSoftmax(dim=1)(x)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GCN(hidden_channels=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(64):
    total_loss = 0
    
    for batch in dataloader:
        if batch is None:
#             print("None")
            continue  # Skip this iteration if the batch is None
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        _, preds = torch.max(out, dim=1)  # Get the predicted class labels
        loss = criterion(out, batch.y)
#         print("Loss: ", loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch}, Average Loss: {avg_loss}')


# Save the trained model
torch.save(model.state_dict(), 'gcn_model-otsu.pth')
