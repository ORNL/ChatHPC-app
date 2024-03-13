#!/usr/bin/env bash
set -e

# Test help functions
echo '*** Test -h for scripts ***'

echo python-project-template-add1 -h
python-project-template-add1 -h

# Test default arguments
echo
echo '*** Test basic script functionality ***'

echo python-project-template-add1 1.0
python-project-template-add1 1.0
