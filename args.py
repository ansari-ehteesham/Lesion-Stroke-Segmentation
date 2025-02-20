import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train or Use a Pre-Trained U-Net Model")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "pretrained"],
        default="train",
        help="Choose 'train' to train a new model or 'pretrained' to load an existing model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the saved model (used for 'pretrained' mode or fine-tuning).",
    )
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["new", "fine_tune"],
        default="new",
        help="Choose 'new' to train a completely new model or 'fine_tune' to continue training a pre-trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of Epoch while Training"
    )

    return parser.parse_args() 
