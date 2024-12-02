import argparse


def add1(x: float) -> float:
    """Add 1 to x.

    Args:
        x (float): Input value.

    Returns:
        float: Output value.
    """
    return x + 1.0


def init_parser(parser):
    parser.add_argument("value", metavar="v", type=float, nargs=1)


def main(raw_args=None):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Description""")
    init_parser(parser)
    args = parser.parse_args(raw_args)
    print(f"add1({args.value[0]}) returns: {add1(args.value[0])}")


if __name__ == "__main__":
    main()
