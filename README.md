# Tokio2Chrome
A small tool to convert [tokio] tracing data captured by [perf] to the Google [Trace Event Format] that can be read
using [Google Chrome] / [Chromium] or [Speedscope].


## Why
Linux [perf] is a very powerful tool to understand the performance of an app. While it knows about threads, the [tokio]
tasks are completely opaque to it and hence you'll have a hard time reading the `perf script` output. The tooling in
this repo wants to help with this.

When thinking about [tokio]s scheduling model, I think we should just talk about [green threads] because that's technically what async tasks are. So [tokio] is _just_ a user land scheduler for [green threads]. With the probes (see [Technical Background](#technical-background)) the state diagram of a task looks like this:

```mermaid
flowchart LR
    Start((Start))
    End((End))

    Scheduled["`
**Scheduled**

Wait for tokio scheduler.
`"]

    Polling["`
**Polling**

NO long-blocking OPs!
`"]

    Blocking["`
**Blocking**

Tokio knows that
thread may block.

Syscalls are OK.
`"]

    Sleep["`
**Sleep**

Wait for other async
task/waker.
`"]

    Syscall["`
**Syscall**

Blocking sycall.
`"]

    %% Task creation
    Start -- "`task_start`" --> Sleep

    %% How to get out of sleep
    Sleep -- "`task_schedule_start`" --> Scheduled
    Sleep -- "`task_finish`" --> End

    %% In-out of Pooling
    Scheduled -- "`task_poll_begin`" --> Polling
    Polling -- "`task_poll_end`" --> Sleep

    %% Blocking
    Polling -- "`task_blocking_begin`" --> Blocking
    Blocking -- "`task_blocking_end`" --> Polling

    %% Syscalls
    Polling -- "`sys_enter`" --> Syscall
    Syscall -- "`sys_exit`" --> Polling
```

## Technical Background
Ideally this would work without modifying the source code and just by using dynamic tracepoints. Seems
[we cannot have nice things](https://lore.kernel.org/linux-perf-users/YboC1QIP342BBz5t@kernel.org/) though. I unable to
use `perf probe` to insert tracepoints at the right places, because `perf` did neither accept the demangled Rust
symbols nor the mangled ones.

So have modified [tokio] to include [SystemTap SDT] probes and provide some tooling to process these.

Here are two example outputs:

![Output Chrome](img/chrome.png)

![Output Catapult](img/catapult.png)

## Usage

### Test App
We will use a modified version of [InfluxDB IOx]. Start by creating a clean directory:

```console
$ mkdir src
$ cd src
```

Get [patched version of tokio](https://github.com/crepererum/tokio/tree/crepererum/probes):

```console
$ git clone -b crepererum/probes https://github.com/crepererum/tokio.git
```

Get [patched version of InfluxDB IOx](https://github.com/influxdata/influxdb_iox/tree/crepererum/probes):

```console
$ git clone -b crepererum/probes https://github.com/influxdata/influxdb_iox.git
```

Build the app:

```console
$ cargo build --profile=quick-release
```

Finally, register the app tracepoints:

```console
$ sudo ./setup_iox.sh
```

### User Config (optional)
To improve `perf` output, run:

```console
$ ./setup_user.sh
```

### System Setup
You won't have the permissions to just run `perf record`, so we need to weaken the system a bit:

**IMPORTANT: Run this AFTER `setup_iox.sh`!**

```console
$ sudo ./setup_system.sh
```

(sometimes this will fail on the first run, just run it a second time)

### Record

```console
$ ./record.sh target/quick-release/influxdb_iox all-in-one ...
```

### Analyze
First convert the binary output (`perf.data`) to text dump (`perf.txt`):

```console
$ perf script > perf.txt
```

Then convert it to a [JSON]:

```console
$ ./script2chrome.py
```

You can load the resulting `perf.json` into [Google Chrome] / [Chromium] or [Speedscope].

You can also use [Catapult] to convert `perf.json` into an HTML:

```console
$ path/to/catapult/tracing/bin/trace2html perf.json --output=perf.html --config full
```

### Online Analyzer
Analyzing data offline can be super slow due to the Python script but also because loading large profiles into [Chromium] is not a good UX. A totally different approach is [bpftrace]. For that, just fire up IOx, figure out its PID and then run

```console
$ sudo bpftrace ./online_analyzer.bt <PID>
```

That should print some statistics without the need to gather large quantities of data.


## License

Licensed under either of these:

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)


### Contributing

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in the work, as defined
in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.


[bpftrace]: https://github.com/bpftrace/bpftrace/
[Catapult]: https://github.com/catapult-project/catapult
[Chromium]: https://www.chromium.org/Home/
[Google Chrome]: https://www.google.com/chrome/index.html
[green threads]: https://en.wikipedia.org/wiki/Green_thread
[InfluxDB IOx]: https://github.com/influxdata/influxdb_iox/
[JSON]: https://www.json.org/
[perf]: https://perf.wiki.kernel.org/index.php/Main_Page
[Speedscope]: https://www.speedscope.app/
[SystemTap SDT]: https://sourceware.org/systemtap/wiki/AddingUserSpaceProbingToApps
[tokio]: https://tokio.rs/
[Trace Event Format]: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
