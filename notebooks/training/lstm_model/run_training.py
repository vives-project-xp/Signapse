"""
This script trains a PyTorch LSTM model for sequence-based sign language gesture recognition.

It loads the recorded sequence data, creates the LSTM model, trains it, 
evaluates its performance, and saves the trained model to a file.
"""
import sys
import argparse
from data_utils import *
from train_utils import *
from model_utils import *


def main(args: list[str]) -> None:
    """
    The main function for training the LSTM gesture recognition model.

    Args:
        args (list[str]): A list of command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50) # Increased epochs for better sequence learning
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for the LSTM layer.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked LSTM layers.")
    parser.add_argument("--output", type=str, default="lstm_model.pth")
    parsed_args = parser.parse_args(args)

    print("--- 1. Data Loading and Preprocessing ---")
    # Load sequence dataset using the LSTM-specific function
    # It automatically gets the DATA_PATH from data_utils.py's configuration
    dataset = load_dataset_for_lstm()
    
    # Check if any data was loaded
    if len(dataset) == 0:
        print(f"Error: No sequences loaded. Check your data structure at {DATA_PATH}.")
        sys.exit(1)

    train_dataset, val_dataset = split_dataset(
        dataset, val_ratio=0.2, random_seed=42)
    train_loader, val_loader = get_loaders(
        train_dataset, val_dataset, batch_size=parsed_args.batch_size)
    
    classes = dataset.classes # Retrieve class list from the loaded dataset
    print(f"Total sequences loaded: {len(dataset)}")
    print(f"Training sequences: {len(train_dataset)}, Validation sequences: {len(val_dataset)}")
    print(f"Fixed Sequence Length (T_max): {SEQUENCE_LENGTH}") # Added printout
    print(f"Classes: {classes}")

    print("\n--- 2. Model Creation ---")
    # Calculate the total input feature dimension from the captured data.
    # POSE (33*4) + LH (21*3) + RH (21*3) = 132 + 63 + 63 = 258
    in_dim = 258
    num_classes = len(classes)
    
    # Pass LSTM-specific parameters to the create_model function
    model = create_model(
        num_classes=num_classes, 
        in_dim=in_dim,
        hidden_size=parsed_args.hidden_size,
        num_layers=parsed_args.num_layers
    )
    print(model)
    print(f"Model created. Input Dim: {in_dim}, Hidden Size: {parsed_args.hidden_size}, Layers: {parsed_args.num_layers}")

    print("\n--- 3. Training ---")
    train_model(
        model, 
        train_loader,
        epochs=parsed_args.epochs, 
        lr=parsed_args.lr
    )

    print("\n--- 4. Evaluation and Saving ---")
    # Evaluate model
    accuracy = evaluate_model(model, val_loader)
    print(f"Final Validation Accuracy: {accuracy:.2f}%")

    # Save model
    save_model(model, path=parsed_args.output)
    print(f"Model saved as {parsed_args.output} to local and package directories.")


if __name__ == "__main__":
    main(sys.argv[1:])