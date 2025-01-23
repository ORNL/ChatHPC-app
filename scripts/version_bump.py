#!/usr/bin/env python3

"""Script to bump the version number of the project and release a new version."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import subprocess
import sys
import traceback
from subprocess import check_output

from pytz import timezone

GIT_ROOT = check_output("git rev-parse --show-toplevel", shell=True).decode().strip()  # noqa S602
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TZ = timezone("US/Eastern")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def run(command, verbose=False, noop=False, directory=None):
    """Print command then run command"""
    return_val = ""

    if directory is not None:
        with pushd(directory):
            return run(command, verbose, noop)

    if verbose:
        print(command)
    if not noop:
        try:
            return_val = subprocess.check_output(command, shell=True, stderr=subprocess.PIPE).decode()  # noqa: S602
        except subprocess.CalledProcessError as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}\n\n{e.returncode}\n\n{e.stdout.decode()}\n\n{e.stderr.decode()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise
        except Exception as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise
        if verbose and return_val:
            print(return_val)

    return return_val


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""
    import os
    import subprocess

    pipe = subprocess.Popen(f"bash -c 'source {script} > /dev/null; env'", stdout=subprocess.PIPE, shell=True)  # noqa: S602
    output = pipe.communicate()[0].decode()
    env = dict(line.split("=", 1) for line in output.splitlines())
    os.environ.update(env)


class Version:
    """Class to hold a version."""

    def __init__(self, year: int, month: int, revision: int):
        self.year: int = year
        self.month: int = month
        self.revision: int = revision

    def __str__(self):
        return f"{self.year}.{self.month}.{self.revision}"

    @classmethod
    def from_str(cls, version_string: str):
        """Convert from String."""
        year, month, revision = version_string.split(".")
        return cls(year=int(year), month=int(month), revision=int(revision))

    @classmethod
    def from_dict(cls, dict_obj):
        """Convert from dictionary."""
        return cls(**dict_obj)


def get_next_version(version_override: str | None = None, use_last_commit_date: bool = False) -> Version | str:
    """Return the next version after the current version.

    Args:
        version_override (str): Version override to use instead of the next version.

    Returns:
        str: The next version to use.
    """
    if version_override is not None:
        return version_override

    # Get the date component
    if use_last_commit_date:
        # Get the year via git without the centary and not 0 padding
        year = int(run("git log -1 --format=%cd --date=local --date=format:'%Y'").strip()[-2:])
        # Get the month via git without 0 padding
        month = int(run("git log -1 --format=%cd --date=local --date=format:'%m'"))
    else:
        # Get the year without the centary and not 0 padding
        year = int(str(datetime.datetime.now(tz=TZ).year)[-2:])
        # Get the month without 0 padding
        month = datetime.datetime.now(tz=TZ).month

    # Get the current version
    try:
        current_version = Version.from_str(run("hatch version"))
    except ValueError:
        current_version = None

    # Get current tags
    tags = run("git tag")

    # Get the revision number
    revision = 0

    # If there is a current version and the year and month match, revision is +1.
    if current_version is not None and current_version.year == year and current_version.month == month:
        revision = current_version.revision + 1

    # Find the next available revision number which doesn't match a tag
    while f"{year}.{month}.{revision}" in tags:
        revision += 1

    return Version(year=year, month=month, revision=revision)


def update_changelog(version: Version, filename: str):
    """Update the changelog on version bump.

    Args:
        version (Version): version
        filename (str): Changelog filename.
    """

    # Read the file.
    with open(filename) as fd:
        changelog = fd.readlines()

    # Find the insertion point.
    try:
        unreleased_loc = changelog.index("## [Unreleased]\n")
    except ValueError as e:
        print(f"Error: Malformed CHANGELOG.md. {e}", file=sys.stderr)
        raise

    # Insert the new version
    changelog.insert(unreleased_loc + 1, f'\n## [{version}] - {datetime.datetime.now(tz=TZ).strftime("%Y-%m-%d")}\n')

    # Write the file
    with open(filename, "w") as fd:
        fd.writelines(changelog)


def init_parser(parser):
    parser.add_argument("--version_override", help="Override to the desired version.", type=str, default=None)
    parser.add_argument(
        "--use-last-commit-date",
        dest="use_last_commit_date",
        help="Use the last commit date for the version date.",
        action="store_true",
    )
    parser.add_argument(
        "--no-use-last-commit-date",
        dest="use_last_commit_date",
        help="Use the current date for version date (default).",
        action="store_false",
    )
    parser.set_defaults(use_last_commit_date=False)
    parser.add_argument("--debug", action="store_true", help="Open debug port (5678).")


def main(raw_args=None):
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="""Script to bump the version number of the project and release a new version."""
    )
    init_parser(parser)
    args = parser.parse_args(raw_args)

    if args.debug:
        import debugpy  # noqa: T100

        debugpy.listen(5678)  # noqa: T100
        print("Attach debugger to continue.")
        debugpy.wait_for_client()  # noqa: T100

    # Verify git state and return warning.
    git_state = run("git status --porcelain")
    if git_state != "":
        print(f"Warning: Git state is unclean.\n{git_state}")
        answer = str2bool(input("Continue? "))
        if answer is False:
            sys.exit(1)

    version = get_next_version(args.version_override, args.use_last_commit_date)

    # Update the version
    print("Updating python project version.")
    run(f"hatch version {version}", verbose=True)

    # Update the CHANGELOG.md
    print("Updating CHANGELOG.md")
    update_changelog(version, "CHANGELOG.md")

    # Commit Change.
    print("Committing change.")
    run("git add src/*/__about__.py CHANGELOG.md", verbose=True)
    run(f'git commit -m "Release version {version}."', verbose=True)
    run(f'git tag -a v{version} -m "Version {version}"', verbose=True)

    # Instruct to push to remote.
    branch_name = run("git rev-parse --abbrev-ref HEAD").strip()
    print(f"\nPush tag to remote with: `git push origin {branch_name} {version}`.")


if __name__ == "__main__":
    main()
