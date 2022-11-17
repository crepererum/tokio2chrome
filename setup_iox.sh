#!/usr/bin/env bash

set -eux -o pipefail

readonly IOX_PATH="target/quick-release/influxdb_iox"

# remove potentially old binary
perf buildid-cache --remove "$IOX_PATH"

# add current binary
perf buildid-cache --add "$IOX_PATH"

# register probes
perf probe -v sdt_tokio:task_poll_begin
perf probe -v sdt_tokio:task_poll_end
