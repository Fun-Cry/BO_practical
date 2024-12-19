import torch
import torch.nn as nn

class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim, poly_degree=2, learning_rate=1e-2, epochs=500):
        super().__init__()
        self.input_dim = input_dim
        self.poly_degree = poly_degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Create polynomial features layer
        features_out = self.compute_poly_features_dim()
        self.linear = nn.Linear(features_out, 1)
        
    def compute_poly_features_dim(self):
        # Compute number of polynomial features
        from scipy.special import comb
        return int(sum(comb(self.input_dim + i - 1, i) for i in range(self.poly_degree + 1)))
    
    def poly_features(self, X):
        # Generate polynomial features
        features = [X]
        for degree in range(2, self.poly_degree + 1):
            features.append(torch.pow(X.unsqueeze(2), degree))
        return torch.cat([f.reshape(X.shape[0], -1) for f in features], dim=1)
    
    def fit(self, X, y):
        # Convert to tensors if needed
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            X_poly = self.poly_features(X)
            y_pred = self.linear(X_poly)
            
            # Compute loss
            loss = criterion(y_pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        
        # After fitting, detach parameters from computation graph
        with torch.no_grad():
            for param in self.parameters():
                param.requires_grad = False
    
    def predict(self, X):
        # Convert to tensor if needed
        X = torch.tensor(X, dtype=torch.float32)
        
        # Generate polynomial features and predict
        X_poly = self.poly_features(X)
        return self.linear(X_poly)