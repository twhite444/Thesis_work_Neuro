import pandas as pd
import matplotlib.pyplot as plt
import os

def load_losses(file_path):
    """Load training and validation losses from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Loss file not found: {file_path}")
    
    # Load the CSV file into a DataFrame
    losses = pd.read_csv(file_path)
    return losses

def plot_losses(losses):
    """Plot training and validation losses for each model."""
    models = losses['model'].unique()
    
    plt.figure(figsize=(12, 8))
    
    for model in models:
        model_losses = losses[losses['model'] == model]
        plt.plot(model_losses['fold'], model_losses['train_loss'], label=f'{model} - Train Loss', marker='o')
        plt.plot(model_losses['fold'], model_losses['val_loss'], label=f'{model} - Val Loss', marker='x')
    
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses by Fold')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Path to the losses CSV file
    loss_file_path = 'models/training_validation_losses.csv'  # Update this path if needed

    # Load the losses
    losses = load_losses(loss_file_path)

    # Plot the losses
    plot_losses(losses)

if __name__ == "__main__":
    main()