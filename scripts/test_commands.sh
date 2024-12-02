#!/usr/bin/env bash
set -e

# Test help functions
echo '*** Test -h for scripts ***'

echo ChatKokkos-add1 -h
ChatKokkos-add1 -h

# Test default arguments
echo
echo '*** Test basic script functionality ***'

echo ChatKokkos-add1 1.0
ChatKokkos-add1 1.0
