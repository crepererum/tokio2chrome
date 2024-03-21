#!/usr/bin/env bash

set -eu -o pipefail

exec perf record \
    --call-graph=fp \
    --event=cycles/freq=999/ \
    --event=instructions/freq=999/ \
    --event=raw_syscalls:sys_enter \
    --event=raw_syscalls:sys_exit \
    --event=sdt_tokio:task_blocking_begin \
    --event=sdt_tokio:task_blocking_end \
    --event=sdt_tokio:task_finish \
    --event=sdt_tokio:task_poll_begin \
    --event=sdt_tokio:task_poll_end \
    --event=sdt_tokio:task_schedule_start \
    --event=sdt_tokio:task_start \
    --mmap-pages=256M \
    --sample-cpu \
    --stat \
    --timestamp \
    -- "$@"
