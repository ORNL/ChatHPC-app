import click

from chatkokkos.app import App

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        config()


@cli.command(help="Print current config")
def config():
    app = App()
    app.print_config()


@cli.command(help="Finetune the model.")
def train():
    app = App()
    app.load_base_model()
    app.load_datasets()
    app.tokenize_training_set()
    app.train()


@cli.command(help="Interact with the base model.")
def run_base():
    app = App()
    app.load_base_model()
    app.interactive("base")


def _run_fine():
    app = App()
    app.load_finetuned_model()
    app.interactive()


@cli.command(help="Interact with the finetuned model.")
def run_fine():
    _run_fine()


@cli.command(help="Interact with the merged model.")
def run_merged():
    app = App()
    app.load_merged_model()
    app.interactive("merged")


@cli.command(help="Interact with the model.")
def run():
    _run_fine()


if __name__ == "__main__":
    cli()
