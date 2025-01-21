#!/usr/bin/env bash
set -e

# Test help functions
echo '*** Test -h for scripts ***'

echo ChatKokkos -h
ChatKokkos -h
echo
echo chatkokkos -h
chatkokkos -h

# Test default arguments
echo
echo '*** Test basic script functionality ***'

echo ChatKokkos config
ChatKokkos config
