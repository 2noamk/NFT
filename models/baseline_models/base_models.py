import torch
import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, horizon):
        super(FC, self).__init__()
        self.horizon = horizon
        self.fc1 = nn.Linear(input_dim, output_dim)  # Input layer
        self.fc2 = nn.Linear(output_dim, output_dim)  # Hidden layer
        self.fc3 = nn.Linear(output_dim, output_dim)  # Hidden layer
        self.fc4 = nn.Linear(output_dim, output_dim)  # Hidden layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x
   
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class LSTM(nn.Module):
    def __init__(self, 
                 num_vars, 
                 lookback, 
                 horizon, 
                 hidden_dim, 
                 num_layers
                 ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(num_vars, hidden_dim, num_layers, batch_first=True) #shape torch.Size([batch, lookback, hidden])

        self.linear1 = nn.Linear(hidden_dim, num_vars) #shape torch.Size([batch, lookback, num_vars])
        
        self.linear2 = nn.Linear(lookback, horizon) #shape torch.Size([batch, horizon, num_vars])

    def forward(self, x):
        device = x.device  # Get the device of the input data

        # Initialize hidden and cell states and move them to the correct device
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()

        # print("x", x.shape)
        # print("h0", h0.shape)
        # print("c0", c0.shape)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # print("out", out.shape)
        
        out = self.linear1(out)
        # print("out", out.shape)
        
        out = self.linear2(out.transpose(2,1)).transpose(2,1)
        # print("out", out.shape)
        
        return out
   
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class CNNBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 lookback, 
                 kernel_size, 
                 stride=1, 
                 padding=1, 
                 dropout=0.2):
        super(CNNBlock, self).__init__()
        def out_dim(input_size):
            return int(((input_size + padding - 1) / stride) + 1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.dropout1, nn.ReLU(),
            self.conv2, self.dropout2, nn.ReLU()
        )
        
        out2_dim = out_dim(out_dim(lookback))
        self.adjust_dim = nn.Linear(lookback-2, lookback)

    def forward(self, x):
        # print("Input shape:", x.shape) #Input shape: torch.Size([32, 5, 15])
        out1 = self.conv1(x)
        # print("After conv1:", out1.shape) # After conv1: torch.Size([32, 25, 16])
        
        out1 = self.dropout1(out1)
        out1 = nn.ReLU()(out1)

        out2 = self.conv2(out1)
        # print("After conv2:", out2.shape) # After conv2: torch.Size([32, 25, 17])
        
        out2 = self.dropout2(out2)
        # out2 = nn.ReLU()(out2)
        
        output = self.adjust_dim(out2)
        # print("After adjust_dim:", out2.shape)
        
        # output = self.adjust_dim2(output.transpose(1, 2)).transpose(1, 2)
        # print("After adjust_dim:", output.shape)
        return output


class CNN(nn.Module):
    def __init__(self, 
                 lookback, 
                 horizon, 
                 num_vars,
                 num_channels=[25, 50], 
                 kernel_size=2, 
                 dropout=0.2):
        super(CNN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = lookback if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * 0
            layers += [CNNBlock(in_channels, 
                                out_channels, 
                                num_vars, 
                                kernel_size=kernel_size, 
                                padding=padding,
                                dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear1 = nn.Linear(num_channels[-1], horizon)

    def forward(self, x):
        # print("x", x.shape)
        y1 = self.network(x)
        # print("y1", y1.shape)
        z1 =  self.linear1(y1.transpose(1, 2)).transpose(1, 2)
        # print("z1", z1.shape)
        return z1
    
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 lookback, 
                 horizon, 
                 num_vars,
                 num_layers, 
                 num_heads, 
                 dim_feedforward, 
                 dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Linear(num_vars, dim_feedforward)
        self.transformer = nn.Transformer(
            d_model=dim_feedforward,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.output_layer1 = nn.Linear(dim_feedforward, num_vars)
        self.output_layer2 = nn.Linear(lookback, horizon)

    def forward(self, src):
        # print(src.shape)
        # Embedding input
        src = self.embedding(src)
        # print(src.shape)
        # Transformer requires a batch first input to be converted to sequence first
        src = src.permute(1, 0, 2)
        # print(src.shape)

        # Pass through the transformer
        output = self.transformer(src, src)
        # print(output.shape)
        # Convert back to batch first
        output = output.permute(1, 0, 2)
        # print(output.shape)

        # Pass through the output layer
        output = self.output_layer1(output)
        # print(output.shape)
        output = self.output_layer2(output.transpose(2,1)).transpose(2,1)
        # print(output.shape)
        return output
   
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class TemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_vars, 
                 kernel_size, 
                 stride=1, 
                 dilation=1, 
                 padding=1, 
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        def out_dim(input_size):
            return int(((input_size + padding - 1) / stride) + 1)
        
        self.conv1 = nn.Conv1d(in_channels, 
                               out_channels, 
                               kernel_size,
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, 
                               out_channels, 
                               kernel_size,
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.dropout1, nn.ReLU(),
            self.conv2, self.dropout2, nn.ReLU()
        )
        
        out2_dim = out_dim(out_dim(num_vars))
        self.adjust_dim = nn.Linear(out2_dim, num_vars)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.dropout1(out1)
        out1 = nn.ReLU()(out1)
        out2 = self.conv2(out1)
        out2 = self.dropout2(out2)
        out2 = nn.ReLU()(out2)
        res = x if self.downsample is None else self.downsample(x)
        out2 = self.adjust_dim(out2)
        output = self.relu(out2 + res)
        return output
    

class TCN(nn.Module):
    def __init__(self, 
                 lookback, 
                 horizon, 
                 num_vars, 
                 num_channels=[25, 50], 
                 kernel_size=2, 
                 dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = lookback if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers += [TemporalBlock(in_channels, 
                                     out_channels, 
                                     num_vars, 
                                     kernel_size=kernel_size, 
                                     dilation=dilation, 
                                     padding=padding, 
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear1 = nn.Linear(num_channels[-1], horizon)

    def forward(self, x):
        y1 = self.network(x)
        z1 =  self.linear1(y1.transpose(1, 2)).transpose(1, 2)
        return z1
    
    def predict(self, x):
        """
        Predict the output given input data x.
        :param x: Input data
        :return: Model's predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions
