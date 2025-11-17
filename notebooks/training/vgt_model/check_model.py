from model_utils import *
import argparse
import sys

def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="vgt_alphabet_model.pth",
        help="Path to the saved model file",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=26,
        help="Number of classes for the model",
    )
    parser.add_argument(
        "--in_dim",
        type=int,
        default=63,
        help="Input dimension for the model",
    )
    parsed_args = parser.parse_args(args)

    model = create_model(num_classes=parsed_args.num_classes, in_dim=parsed_args.in_dim)
    model = load_model(
        path=parsed_args.model_path,
        model=model,
    )
    print("Model loaded successfully:")
    print(model)


if __name__ == "__main__":
    main(sys.argv[1:])
