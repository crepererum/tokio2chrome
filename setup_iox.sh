#!/usr/bin/env bash

set -eux -o pipefail

readonly IOX_PATH="target/quick-release/influxdb_iox"

# remove potentially old binary
perf buildid-cache --remove "$IOX_PATH"

# add current binary
perf buildid-cache --add "$IOX_PATH"

# register probes
probes=( task_finish task_poll_begin task_poll_end task_blocking_begin task_blocking_end )
for probe in "${probes[@]}"; do
    set +e
    perf probe --del="sdt_tokio:$probe"
    set -e

    perf probe --verbose --add="sdt_tokio:$probe"
done
