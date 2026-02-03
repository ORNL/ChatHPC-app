#!/bin/bash

# Convert SSH git URLs to HTTPS in pyproject.toml
sed -i 's|git+ssh://git@github.com/|git+https://github.com/|g' pyproject.toml
