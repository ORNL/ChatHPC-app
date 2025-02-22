import argparse
import sys

from loguru import logger
from pydantic import ValidationError
from pydantic_settings import (
    CliApp,
    CliSettingsSource,
)

from chathpc.app import App, AppConfig
from chathpc.app.utils.common_utils import load_json_arg


def config(_args, config):
    app = App(config)
    app.print_config()


def train(_args, config):
    app = App(config)
    app.print_config()
    app.load_base_model()
    app.load_datasets()
    app.tokenize_training_set()
    app.train()


def run_base(args, config):
    app = App(config)
    app.load_base_model()
    app.interactive(args, "base")


def _run_fine(args, config):
    app = App(config)
    app.load_finetuned_model()
    app.interactive(args)


def run_fine(args, config):
    _run_fine(args, config)


def run_merged(args, config):
    app = App(config)
    app.load_merged_model()
    app.interactive(args, "merged")


def run(args, config):
    _run_fine(args, config)


def init_parser(parser):
    parser.add_argument("--debug", action="store_true", help="Open debug port (5678).")
    parser.add_argument("--log_level", type=str, help="Log level.")
    parser.add_argument("--config", type=str, help="Path to config json file.")
    parser.add_argument(
        "--extract",
        dest="extract",
        action="store_true",
        help="Extract the answer from the response.",
    )
    parser.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        help="Show the response without extracting the answer.",
    )
    parser.set_defaults(extract=False)
    parser.set_defaults(func=config)

    subparsers = parser.add_subparsers(title="subcommands", description="valid subcommands")

    subparsers.add_parser("config", help="Print current config").set_defaults(func=config)
    subparsers.add_parser("train", help="Finetune the model").set_defaults(func=train)
    subparsers.add_parser("run", help="Interact with the model").set_defaults(func=run)
    subparsers.add_parser("run_base", help="Interact with the base model").set_defaults(func=run_base)
    subparsers.add_parser("run_fine", help="Interact with the finetuned model").set_defaults(func=run_fine)
    subparsers.add_parser("run_merged", help="Interact with the merged model").set_defaults(func=run_merged)


def cli(raw_args=None):
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="""ChatHPC Application. Used to train and interact with ChatX applications which are part of ChatHPC."""
    )
    init_parser(parser)

    # Set existing `parser` as the `root_parser` object for the user defined settings source
    cli_settings = CliSettingsSource(AppConfig, root_parser=parser)

    # Parse and load AppConfig settings from the command line into the settings source.
    args = parser.parse_args(raw_args)

    # Setup debugging.
    if args.debug:
        if args.log_level is None:
            args.log_level = "DEBUG"

        import debugpy  # noqa: T100

        debugpy.listen(5678)  # noqa: T100
        print("Attach debugger to continue.")
        debugpy.wait_for_client()  # noqa: T100

    # Setup log level
    if args.log_level:
        logger.remove()
        logger.add(sys.stderr, level=args.log_level.upper())
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Read in configuration file.
    try:
        json_config = load_json_arg(args.config)
        app_config = CliApp.run(AppConfig, cli_args=args, cli_settings_source=cli_settings, **json_config)
    except ValidationError as e:
        parser.print_help()
        print()
        print(e)
        sys.exit(1)

    # Run the subcommand.
    try:
        args.func(args, app_config)
    except EOFError:
        print("\nGoodbye!")


if __name__ == "__main__":
    cli()
