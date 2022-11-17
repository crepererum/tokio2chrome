#!/usr/bin/env bash

set -eux -o pipefail

perf config record.debuginfod=system
