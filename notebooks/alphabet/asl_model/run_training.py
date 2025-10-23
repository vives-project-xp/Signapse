from data_utils import *
from train_utils import *
from model_utils import *
import argparse
import sys

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="hand_gesture_model.pth")
    parsed_args = parser.parse_args(args)

    # Load data
    classes = get_classes()
    dataset = load_and_preprocess_dataset(HAND_LANDMARKS_CSV)
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=0.2, random_seed=42)
    train_loader, val_loader = get_loaders(train_dataset, val_dataset, batch_size=parsed_args.batch_size)

    # Create model
    in_dim = 63  # 21 landmarks * 3 coordinates (x, y, z)
    num_classes = len(classes)
    model = create_model(num_classes, in_dim)

    # Train model
    train_model(model, train_loader, epochs=parsed_args.epochs, lr=parsed_args.lr)

    # Evaluate model
    accuracy = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save model
    save_model(model, path=parsed_args.output)
    print(f"Model saved to {parsed_args.output}")

if __name__ == "__main__":
    main(sys.argv[1:])