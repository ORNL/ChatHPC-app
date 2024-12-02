# ChatKokkos

For general information on packaging python projects see [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/).

## Tools Used

- [Hatch](https://hatch.pypa.io/) --- Python Build System.
- [MkDocs](https://www.mkdocs.org/) --- Documentation Generator.
    - [Material Theme](https://squidfunk.github.io/mkdocs-material/) --- Theme for documentation.
    - [mkdocstrings](https://mkdocstrings.github.io/) --- Automatic documentation generation from sources.
    - [DevDocs](https://docs.excl.ornl.gov/quick-start-guides/devdocs) --- Internal to ORNL document website hosting.
- [GitLab CI](https://docs.gitlab.com/ee/ci/) --- Continuous Integration.
    - [Example Pipeline](https://code.ornl.gov/ChatHPC/ChatKokkos/-/pipelines)
    - [Example Pipeline Source](https://code.ornl.gov/ChatHPC/ChatKokkos/-/blob/main/.gitlab-ci.yml?ref_type=heads)
- [Ruff](https://docs.astral.sh/ruff/) --- Python linter and code formater.
    - [Ruff Rules](https://docs.astral.sh/ruff/rules/) --- Rules used by Ruff.
- [EditorConfig](https://editorconfig.org/) --- Maintain consistent coding styles between different editors and IDEs.
- [Markdown Lint Tool](https://github.com/markdownlint/markdownlint) --- Markdown linting tool.
- [Pre-Commit](https://pre-commit.com/) --- Git precommit hooks.
    - [Built-in Hooks](https://github.com/pre-commit/pre-commit-hooks)
    - [Ruff Pre-Commit Hooks](https://github.com/astral-sh/ruff-pre-commit)
    - [Editor Config Pre-Commit Hooks](https://github.com/editorconfig-checker/editorconfig-checker.python)
    - [Markdown Lint Pre-Commit Hooks](https://github.com/markdownlint/markdownlint)

## Quick Start with this Template

1. Fork the repository.
2. Run `setup_template.sh` to setup the template for the new project.
3. Remove `setup_template.sh`

## Steps to Manually Setup Hatch and MkDocs Python Repo with CI Setup

This repo was setup by following this general process.

### Hatch Setup

```bash
pip install hatch
hatch new "Project Name"
```

Customize `pyproject.toml`.

Add:
```toml
[tool.hatch.metadata]
allow-direct-references = true
```

### Lint Rules

<!-- editorconfig-checker-disable -->
```toml
[tool.hatch.envs.hatch-static-analysis]
config-path = "ruff_defaults.toml"

[tool.ruff]
extend = "ruff_defaults.toml"

[tool.ruff.lint]
extend-ignore = [
  "T201", # `print` found
  "FBT001", # Boolean-typed positional argument in function definition
  "FBT002", # Boolean default positional argument in function definition
]
```
<!-- editorconfig-checker-enable -->

Generate default ruff rules.
```bash
hatch fmt --check --sync
```

### MkDocs Setup

```bash
pip install mkdocs
mkdocs new .
```

Customize `mkdocs.yml`.

Example:
<!-- editorconfig-checker-disable -->
```yml
site_name: Example
nav:
  - index.md
  - api.md

theme:
  name: material

plugins:
  - search
  - autorefs
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [src]

markdown_extensions:
  # Built-in
  - markdown.extensions.abbr:
  - markdown.extensions.admonition:
  - markdown.extensions.attr_list:
  - markdown.extensions.footnotes:
  - markdown.extensions.md_in_html:
  - markdown.extensions.meta:
  - markdown.extensions.tables:
  - markdown.extensions.toc:
      permalink: true
```
<!-- editorconfig-checker-enable -->

Add the following to `pyproject.toml`.
<!-- editorconfig-checker-disable -->
```toml
[tool.hatch.env]
requires = [
  "hatch-mkdocs",
]

[tool.hatch.env.collectors.mkdocs.docs]
path = "mkdocs.yml"

[tool.hatch.envs.docs]
```
<!-- editorconfig-checker-enable -->

Update the coverage section:
<!-- editorconfig-checker-disable -->
```toml
[tool.hatch.envs.default.scripts]
test = "pytest {args:tests}"
test-cov = "coverage run -m pytest {args:tests}"
cov-report = [
  "- coverage combine",
  "coverage report",
  "coverage xml -o coverage.xml",
]
cov-html = [
  "test-cov",
  "cov-report",
  "coverage html",
]
cov = [
  "test-cov",
  "cov-report",
]

[tool.coverage.report]
exclude_lines = [
  "no cov",
  "if __name__ == .__main__.:",
  "if TYPE_CHECKING:",
  "raise AssertionError",
  "raise RuntimeError",
  "raise NotImplementedError",
  "pass",
  "raise ValueError",
]

[tool.coverage.html]
directory = "coverage_html_report"
```
<!-- editorconfig-checker-enable -->

Note: If you project uses a namespace package, you will need to specify the package wheel target.
```bash
[tool.hatch.build.targets.wheel]
packages = ["src/<package_name>"]
```

### Readme Example

Note the readme example's Table of Contents is kept up-to-date using the [Markdown All in One](https://github.com/yzhang-gh/vscode-markdown) VSCode extension.

Example Readme Content:

Documentation: <https://devdocs.ornl.gov/example>  
Coverage Report: <https://devdocs.ornl.gov/example/coverage>

----

**Table of Contents**

- [ChatKokkos](#ChatKokkos)
    - [Tools Used](#tools-used)
    - [Quick Start with this Template](#quick-start-with-this-template)
    - [Steps to Manually Setup Hatch and MkDocs Python Repo with CI Setup](#steps-to-manually-setup-hatch-and-mkdocs-python-repo-with-ci-setup)
        - [Hatch Setup](#hatch-setup)
        - [Lint Rules](#lint-rules)
        - [MkDocs Setup](#mkdocs-setup)
        - [Readme Example](#readme-example)
            - [Installation](#installation)
            - [Setup pre-commit Git hooks](#setup-pre-commit-git-hooks)
            - [Running with hatch](#running-with-hatch)
            - [Testing with hatch](#testing-with-hatch)
            - [Format code with hatch](#format-code-with-hatch)
            - [View version with hatch](#view-version-with-hatch)
            - [Update version with hatch](#update-version-with-hatch)
            - [Documentation](#documentation)
                - [Commands](#commands)
                - [Hatch Commands](#hatch-commands)
        - [Setup CI](#setup-ci)
        - [.gitignore](#gitignore)
        - [Example Changelog](#example-changelog)
            - [Changelog Header Template](#changelog-header-template)
        - [Setup Pre-Commit](#setup-pre-commit)
        - [Setup Editor Config](#setup-editor-config)

#### Installation

For development in folder:

```bash
git clone git@code.ornl.gov:example.git
cd example
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For use in virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install git+ssh://git@code.ornl.gov/example.git
```

#### Setup pre-commit Git hooks

Use hatch or install pre-commit inside python virtual environment.
```bash
hatch shell
```
or
```bash
pip install pre-commit
```

Then install the hooks.
```bash
pre-commit install
```

Note: You might have to upgrade pre-commit.
```bash
pre-commit autoupdate
```

#### Running with hatch

```bash
hatch shell
```

#### Testing with hatch

```bash
hatch run test
```

To test on all python versions:
```bash
hatch run all:test
```

Run tests and print the output.
```bash
hatch run test -v -s
```

#### Format code with hatch

```bash
hatch fmt
```

Update default ruff rules:

```bash
hatch fmt --check --sync
```

#### View version with hatch

```bash
hatch version
```

#### Update version with hatch

```bash
hatch version <new version>
```

#### Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org) using the [Read the Docs](https://docs.readthedocs.io/en/stable/) theme.

##### Commands

- `mkdocs new [dir-name]` - Create a new project.
- `mkdocs serve` - Start the live-reloading docs server.
- `mkdocs build` - Build the documentation site.
- `mkdocs -h` - Print help message and exit.

Other useful commands:
- `mkdocs serve -a 0.0.0.0:8000` - Serve with extenal access to the site. (Useful in ExCL to view using foxyproxy.)

##### Hatch Commands

View environment
```bash
hatch env show docs
```

Build documentation
```bash
hatch run docs:build
```

Serve documentation
```bash
hatch run docs:serve
```
or
```bash
hatch run docs:serve -a 0.0.0.0:8000
```

----

### Setup CI

See [Devdocs](devdocs.md) and [gitlab-ci](gitlab-ci.md).

Example `.gitlab-ci.yml`:

<!-- editorconfig-checker-disable -->
```yml
stages:
  - build
  - check_format
  - docs
  - test
  - check_coverage
  - deploy_coverage
  - deploy_docs

before_script:
  - python3 -m venv venv
  - source venv/bin/activate
  - pip install --upgrade pip

.zenith:
  tags: [ubuntu, zenith]

build-job:
  extends: [.zenith]
  stage: build
  script:
    - pip install .
    - python3 -c "import chatkokkos; print(chatkokkos.__doc__)"

check_format-job:
  extends: [.zenith]
  stage: check_format
  needs: []
  script:
    - pip install hatch
    - pip install --upgrade hatch-mkdocs
    - (hatch fmt | tee fmt.out && echo "lint_errors 0" > metrics.txt) || (cat fmt.out | grep -e 'Found .* errors' | sed 's/Found \(.*\) errors.*/lint_errors \1/' > metrics.txt)
    - cat metrics.txt
    - grep -q "lint_errors 0" metrics.txt
  artifacts:
    paths:
      - fmt.out
      - metrics.txt
    reports:
      metrics: metrics.txt
  allow_failure:
    exit_codes:
      - 1

docs-job:
  tags: [devdocs]
  stage: docs
  needs: []
  script:
    - pip install --upgrade hatch
    - hatch run docs:build
  artifacts:
    paths:
      - site

test-job:
  extends: [.zenith]
  stage: test
  needs: [build-job]
  script:
    - pip install --upgrade hatch
    - hatch env prune # Needed when framework changes since version is not updated.
    - hatch run test

coverage-job:
  extends: [.zenith]
  stage: check_coverage
  needs: [build-job]
  script:
    - pip install --upgrade hatch
    - hatch env prune # Needed when framework changes since version is not updated.
    - hatch run cov-html
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    paths:
      - coverage_html_report
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

.deploy_coverage_common:
  tags: [devdocs]
  stage: deploy_coverage
  needs: [coverage-job]
  script:
    - rsync -a --delete coverage_html_report/ ~/www/ChatHPC/ChatKokkos/coverage

deploy_coverage-job:
  extends: .deploy_coverage_common
  only:
    - main@ChatHPC/ChatKokkos

deploy_coverage_manual-job:
  extends: .deploy_coverage_common
  when: manual
  only:
    - branches@ChatHPC/ChatKokkos

.deploy_docs_common:
  tags: [devdocs]
  stage: deploy_docs
  needs: [docs-job]
  script:
    - rsync -a --delete site/ ~/www/ChatHPC/ChatKokkos

deploy_docs-job:
  extends: .deploy_docs_common
  only:
    - main@ChatHPC/ChatKokkos

deploy_docs_manual-job:
  extends: .deploy_docs_common
  when: manual
  only:
    - branches@ChatHPC/ChatKokkos
```
<!-- editorconfig-checker-enable -->

### .gitignore

```txt
tmp/
*.out
*.swp
*.whl
*.old
*.o
__pycache__/
pyvenv.cfg
finder.txt
.DS_STORE
/.vscode/
venv/
/dist/
/site/
.coverage
coverage.xml
metrics.txt
```

### Example Changelog

#### Changelog Header Template

This is template for a changelog header to explain the purpose and format of the changelog.
The actual log will be added below.

CHANGELOG.md:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to a custom Development Versioning specified by Aaron Young.

A summary of Development Versioning Specification is shown below.

> Given a version number BRANCH.TAG.BUILD, increment the:
> 1. BRANCH version when you make breaking/major changes that you want to track in a separate branch.
> 2. TAG version when you make a new tag to mark a specific spot.
> 3. BUILD version when you create a new build with artifacts or bug fixes for that you want to point to.
>
> Then for your repo you have branch versions for each version. For example branches v0 and v1. Then when you create tags, say on branch v0, you would create tags v0.0.0, v0.1.0, and v0.2.0.
> CI or a manual process could add v0.0.x branches as new changes are added to a local branch. BUILD is also used when patches are applied to a tagged branch, after the patch is applied, add a new tag with BUILD + 1.
>
> `main` always points to the current major branch plus 1. `dev` is an integration branch before merging into `main`. When `dev` is merged into `main`, the TAG is updated.

## [Unreleased]
```

### Setup Pre-Commit

Use `pre-commit sample-config` to generate sample config and then modify it to add the desired hooks.

### Setup Editor Config

Add `.editorconfig` file to the root of the project.  See <https://EditorConfig.org> for details.
