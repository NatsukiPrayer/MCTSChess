import argparse


def parse(*args, **kwargs):
    parser = argparse.ArgumentParser(prog="Chess")

    # TODO: Проверить что аргументы работают
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test"], help="specify run mode: train or test"
    )
    parser.add_argument("--config", type=str, default="configs/config.json", help="path to config file")

    return parser.parse_args()
