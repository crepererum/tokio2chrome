#!/usr/bin/env bash

set -eux -o pipefail

sysctl kernel.perf_event_paranoid=-1
chmod -R 755 /sys/kernel/debug/
chmod -R 755 /sys/kernel/tracing/
