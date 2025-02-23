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

An alternative approach is to use date-based versioning.

With this method, the version is YEAR.MONTH.RELEASE. To increment this version, use the year and the date without 0 padding for the first two numbers. I prefer to use the year without the centary. Then increment the RELEASE number to a unique release. This process is done automatically by the `scripts/version_bump.py` script. Using this script is the prefered method for versioning without planned backporting of fixes.

## [Unreleased]

### Added

- Config: `prompt_template_file`.
- Config: `prompt_template`.
- APP: `chat_evaluate_extract` to chathpc to evaluate and extract the answer portion in one call.
- APP: `save_readme` function to save starter readme for models.
- Train: Save a template readme in the output folders.
- CLI app: log_level is now a command line argument.
- Template: Suport relative paths from the config file.
- Interactive: Added optional ability to extract answer from response with `--extract`.
- Interactive: Gracefully handle EOF.
- Interactive & Template: Make context optional.
- Interactive: Allow context to be added inline with `/context <context>`.
- Interactive: A blank context, unsets the context.
- Ollama: Test both generate and chat API.
- CLI & Method: Verify to verify the model against the training dataset.

### Changed

- Template: Switch from format string to Jinja for the templates.
- Template: Now only one template is used for both training and inference.
- Template: Can either be set by a file path or a string.

## Removed

- Removed `training_prompt`. Replaced by `prompt_template`.
- Removed `inference_prompt`. Replaced by `prompt_template`.

## [25.2.0] - 2025-02-21

### Added

- Added `max_training_tokens` to specify the `max_length` parameter for the tokenizer when tokenizing the training set. Defaults to the prior setting of 512.

### Changed

- Logging: Switch from using logging module to loguru.

### Fixed

- load_json_arg(): Fixed bug with json string as input.
- Fixed bug in interactive run mode.
- Fixed extract_answer() utility to use the prompt and end string to extract the answer portion.

## [25.1.0] - 2025-01-25

Verified initial working version of the ChatHPC App.

### Added

- Initial version of ChatHPC App. Created from the working ChatKokkos Example.

[unreleased]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.2.0...main
[25.2.0]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/compare/v25.1.0...v25.2.0
[25.1.0]: https://code.ornl.gov/ChatHPC/ChatHPC-app/-/releases/v25.1.0
