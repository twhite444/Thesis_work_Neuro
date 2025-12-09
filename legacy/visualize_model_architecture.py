import torch
from model_comparison_pytorch import NeuralNetwork
import netron

if __name__ == "__main__":
    # Example usage
    input_size = (64,)  # Number of features in X
    output_size = 1
    hidden_sizes = [512, 256, 128]  # Deep architecture with 4 layers
    dropout_rate = 0.4  # Optimal dropout rate for regularization

    # Initialize the model
    model = NeuralNetwork(input_size=input_size[0], output_size=output_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)

    # Export the model to ONNX format
    dummy_input = torch.randn(1, *input_size)  # Batch size of 1
    torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])

    # Visualize the model in Netron
    netron.start("model.onnx")