#!/usr/bin/env bash
set -e

# Test help functions
echo '*** Test -h for scripts ***'

echo ChatHPC -h
ChatHPC -h
echo
echo chathpc -h
chathpc -h

# Test default arguments
echo
echo '*** Test basic script functionality ***'

echo chathpc config
chathpc config
