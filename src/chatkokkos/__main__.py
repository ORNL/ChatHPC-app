import click

from chatkokkos.app import App


@click.command()
@click.option("-c", "--config", default=None, help="Config file.")
def main(config=None):
    app = App(config)
    app.print_preferences()


if __name__ == "__main__":
    main()
