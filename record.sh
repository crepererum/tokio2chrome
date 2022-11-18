#!/usr/bin/env bash

set -eu -o pipefail

exec perf record \
    --call-graph=dwarf,65528 \
    --event=cycles/freq=9999/ \
    --event=instructions/freq=9999/ \
    --event=raw_syscalls:sys_enter \
    --event=raw_syscalls:sys_exit \
    --event=sdt_tokio:task_poll_begin \
    --event=sdt_tokio:task_poll_end \
    --sample-cpu \
    --stat \
    --timestamp \
    -- "$@"
