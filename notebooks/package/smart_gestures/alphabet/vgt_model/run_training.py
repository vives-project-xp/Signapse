from .train_utils import train_model, evaluate_model
import argparse
import sys

from .model_utils import create_model, save_model
from .data_utils import load_dataset_normalized, split_dataset, get_loaders, get_classes


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="vgt_model.pth")
    # Augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--rotate_deg", type=float, default=10.0)
    # Scheduler / Early stopping / Checkpoints
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["none", "plateau", "step", "cosine"])
    parser.add_argument("--scheduler_factor", type=float,
                        default=0.5, help="ReduceLROnPlateau factor or ignored")
    parser.add_argument("--scheduler_patience", type=int,
                        default=3, help="ReduceLROnPlateau patience or ignored")
    parser.add_argument("--step_size", type=int, default=10,
                        help="StepLR step_size if scheduler=step")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="StepLR decay factor if scheduler=step")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--monitor", type=str, default="val_loss", choices=[
                        "val_loss", "val_acc"], help="Metric for early stop/checkpoint mode")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Enable model checkpointing")
    parser.add_argument("--checkpoint_path", type=str,
                        default="best_vgt_model.ckpt")
    parsed_args = parser.parse_args(args)

    # Load data
    classes = get_classes()
    dataset = load_dataset_normalized(
        HAND_LANDMARKS_JSON,  # type: ignore  # TODO: define HAND_LANDMARKS_JSON path
        as_sequence=False,
        scale_method="wrist_to_middle",
        augment=parsed_args.augment,
        augment_prob=parsed_args.augment_prob,
        noise_std=parsed_args.noise_std,
        rotate_deg=parsed_args.rotate_deg,
    )
    train_dataset, val_dataset = split_dataset(
        dataset, val_ratio=0.2, random_seed=42)
    train_loader, val_loader = get_loaders(
        train_dataset, val_dataset, batch_size=parsed_args.batch_size)  # type: ignore

    # Create model
    in_dim = 63  # 21 landmarks * 3 coordinates (x, y, z)
    num_classes = len(classes)
    model = create_model(num_classes, in_dim)

    # Prepare callbacks configs
    scheduler_kwargs = {}
    if parsed_args.scheduler == "plateau":
        scheduler_kwargs = {"factor": parsed_args.scheduler_factor,
                            "patience": parsed_args.scheduler_patience}
    elif parsed_args.scheduler == "step":
        scheduler_kwargs = {
            "step_size": parsed_args.step_size, "gamma": parsed_args.gamma}

    early_stopping_kwargs = {
        "patience": parsed_args.early_stop_patience,
        "min_delta": parsed_args.early_stop_min_delta,
        "mode": "min" if parsed_args.monitor == "val_loss" else "max",
        "restore_best_weights": True,
    }
    checkpoint_kwargs = {}
    if parsed_args.checkpoint:
        checkpoint_kwargs = {
            "filepath": parsed_args.checkpoint_path,
            "monitor": parsed_args.monitor,
            "mode": "min" if parsed_args.monitor == "val_loss" else "max",
            "save_best_only": True,
        }

    # Train model
    train_model(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=parsed_args.epochs,
        lr=parsed_args.lr,
        scheduler_type=parsed_args.scheduler,
        scheduler_kwargs=scheduler_kwargs,
        early_stopping_kwargs=early_stopping_kwargs,
        checkpoint_kwargs=checkpoint_kwargs,
    )

    # Evaluate model
    accuracy = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save model
    save_model(model, path=parsed_args.output)
    print(f"Model saved to {parsed_args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
