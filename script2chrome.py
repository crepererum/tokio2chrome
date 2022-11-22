#!/usr/bin/env python3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
import json
import re
from typing import cast, Generator, TypeAlias


PID = 1
VIRT_THREAD_OFFSET = 2**32


# See https://github.com/python/typing/issues/182#issuecomment-1320974824
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

SNAKE_CASE_RE = re.compile(r'(?<!^)(?=[A-Z])')

# See https://stackoverflow.com/a/1176023
def camel2snake(s: str) -> str:
    return SNAKE_CASE_RE.sub('_', s).lower()


class EventDescrType(Enum):
    BEGIN = "B"
    END = "E"
    INSTANT = "i"
    METADATA = "M"


@dataclass
class EventDescrSysEnter:
    nr: int
    ty: EventDescrType = EventDescrType.BEGIN


@dataclass
class EventDescrSysExit:
    nr: int
    ty: EventDescrType = EventDescrType.END


@dataclass
class EventDescrTokioTaskPollBegin:
    task: int
    ty: EventDescrType = EventDescrType.BEGIN


@dataclass
class EventDescrTokioTaskPollEnd:
    task: int
    is_ready: bool
    panicked: bool
    ty: EventDescrType = EventDescrType.END


@dataclass
class EventDescrTokioTaskStart:
    task: int
    ty: EventDescrType = EventDescrType.BEGIN


@dataclass
class EventDescrTokioTaskFinish:
    task: int
    ty: EventDescrType = EventDescrType.END


@dataclass
class EventDescrProcessName:
    name: str
    ty: EventDescrType = EventDescrType.METADATA


@dataclass
class EventDescrProcessSortIndex:
    sort_index: int
    ty: EventDescrType = EventDescrType.METADATA


@dataclass
class EventDescrThreadName:
    name: str
    ty: EventDescrType = EventDescrType.METADATA


@dataclass
class EventDescrThreadSortIndex:
    sort_index: int
    ty: EventDescrType = EventDescrType.METADATA


@dataclass
class EventDescrOther:
    content: str
    ty: EventDescrType = EventDescrType.INSTANT



EventDescr = (
    EventDescrSysEnter
    | EventDescrSysExit
    | EventDescrTokioTaskStart
    | EventDescrTokioTaskFinish
    | EventDescrTokioTaskPollBegin
    | EventDescrTokioTaskPollEnd
    | EventDescrProcessName
    | EventDescrProcessSortIndex
    | EventDescrThreadName
    | EventDescrThreadSortIndex
    | EventDescrOther
)


@dataclass
class EventHeader:
    thread_name: str
    thread_id: int
    core: int
    ts: float
    descr: EventDescr


@dataclass
class Event:
    header: EventHeader
    backtrace: list[str] | None

    def to_chrome(self) -> JSON:
        name = camel2snake(
            type(self.header.descr).__name__.removeprefix("EventDescr").removesuffix("Enter").removesuffix("Exit").removesuffix("Begin").removesuffix("End").removesuffix("Start").removesuffix("Finish")
        )

        out: dict[str, JSON] = {
            "pid": PID,
            "tid": self.header.thread_id,
            # ts is in micros
            "ts": self.header.ts * 1_000_000,
            "cat": name,
            "name": name,
            "ph": self.header.descr.ty.value,
        }

        args = {}
        for k, v in asdict(self.header.descr).items():
            if k == "ty":
                continue
            if isinstance(v, Enum):
                v = v.value
            args[k] = v
        out["args"] = args

        if self.backtrace is not None:
            out["stack"] = cast(JSON, self.backtrace)

        if self.header.descr.ty == EventDescrType.INSTANT:
            out["s"] = "t"

        return out


def extract_syscall_nr(s: str) -> int:
    start = s.find("NR") + 3
    end = start + s[start:].find(" ")
    return int(s[start:end])


def extract_sdt_arg(s: str, argnum: int) -> int:
    arg_txt = f"arg{argnum}="
    start = s.find(arg_txt) + len(arg_txt)
    end_rel = s[start:].find(" ")
    end = start + end_rel if end_rel != -1 else len(s)
    return int(s[start:end])


def parse_descr(s: str) -> EventDescr:
    s = s.strip()

    if "raw_syscalls:sys_enter" in s:
        return EventDescrSysEnter(
            nr=extract_syscall_nr(s),
        )
    elif "raw_syscalls:sys_exit" in s:
        return EventDescrSysExit(
            nr=extract_syscall_nr(s),
        )
    elif "sdt_tokio:task_poll_begin" in s:
        return EventDescrTokioTaskPollBegin(
            task=extract_sdt_arg(s, 1),
        )
    elif "sdt_tokio:task_poll_end" in s:
        return EventDescrTokioTaskPollEnd(
            task=extract_sdt_arg(s, 1),
            is_ready=bool(extract_sdt_arg(s, 2)),
            panicked=bool(extract_sdt_arg(s, 3)),
        )
    elif "sdt_tokio:task_finish" in s:
        return EventDescrTokioTaskFinish(
            task=extract_sdt_arg(s, 1),
        )
    else:
        return EventDescrOther(
            content=s
        )


def parse_header(line: str) -> EventHeader:
    # Unnamed threads are called `:{thread_id}` so we cannot use ":" to split the header line. Split of that colon
    # and put it back later.
    # Example: `:436557 436557 [015] 54146.411072:   sdt_tokio:task_poll_end: (5582cdd22cc2) arg1=139759572301952 arg2=1`
    if line.startswith(":"):
        thread_name_prefix = ":"
        line = line[1:]
    else:
        thread_name_prefix = ""

    # Example: `IOx Query Execu 436743 [005] 54151.787741:          1         cycles/freq=9999/: `
    header_head, header_tail = line.split(":", maxsplit=1)

    # read fields from right to left because thread name may contain spaces
    header_head, ts_str = header_head.rsplit(" ", maxsplit=1)
    ts = float(ts_str)

    header_head, core_str = header_head.rsplit(" ", maxsplit=1)
    assert core_str.startswith("[") and core_str.endswith("]")
    core = int(core_str[1:-1])

    header_head, thread_id_str = header_head.rsplit(" ", maxsplit=1)
    thread_id = int(thread_id_str)

    thread_name = thread_name_prefix + header_head

    descr = parse_descr(header_tail)

    return EventHeader(
        thread_name=thread_name,
        thread_id=thread_id,
        core=core,
        ts=ts,
        descr=descr,
    )


def parse(lines: Iterable[str]) -> Generator[Event, None, None]:
    lines_it = iter(lines)
    line = None

    while True:
        if line is None:
            try:
                line = next(lines_it)
            except StopIteration:
                return

        # looks like header?
        if not line.strip() or line.startswith(" ") or line.startswith("\t"):
            line = None
            continue

        header = parse_header(line)
        backtrace = []
        while True:
            try:
                line = next(lines_it)
            except StopIteration:
                break

            if line.startswith("\t"):
                backtrace.append(line.strip())
            else:
                break

        yield Event(
            header=header,
            backtrace=backtrace,
        )


def metadata_event(name: str, tid: int, args: JSON) -> JSON:
    return {
        "name": name,
        "ph": "M",
        "pid": PID,
        "tid": tid,
        "ts": 0,
        "args": args,
    }


def process_events(events: Iterable[Event]) -> Generator[Event, None, None]:
    events_it = iter(events)

    # ID counter for virtual threads (one for each tokio task)
    virt_thread_counter = 0

    # maps thread ID to virtual thread ID.
    # If a thrad ID is NOT in this dict, use its real ID.
    phys_thread_state: dict[int, int] = {}

    # maps task ID to virtual thread ID and reverse
    known_tasks: dict[int, int] = {}
    known_tasks_rev: dict[int, int] = {}

    # maps threads to names
    thread_names: dict[int, str] = {}

    # flags if we have set up process-level metadata
    process_metadata_done = False

    for evt in events_it:
        task = None
        tid = None
        emit_task_finish = False

        if isinstance(evt.header.descr, EventDescrTokioTaskPollBegin):
            task = evt.header.descr.task

            if evt.header.thread_id in phys_thread_state:
                old_tid = phys_thread_state[evt.header.thread_id]
                old_task = known_tasks_rev[old_tid]
                print(f"already have a task running on this thread: thread={evt.header.thread_id} new_task={task} old_task={old_task}")
                yield Event(
                    header=EventHeader(
                        thread_name=thread_names[old_tid],
                        thread_id=old_tid,
                        core=evt.header.core,
                        ts=evt.header.ts,
                        descr=EventDescrTokioTaskPollEnd(
                            task=old_task,
                            is_ready=False,
                            panicked=False,
                        ),
                    ),
                    backtrace=None,
                )

            tid = known_tasks.get(task)
            if tid is None:
                tid = VIRT_THREAD_OFFSET - virt_thread_counter
                known_tasks[task] = tid
                known_tasks_rev[tid] = task
                virt_thread_counter += 1

            phys_thread_state[evt.header.thread_id] = tid
        elif isinstance(evt.header.descr, EventDescrTokioTaskPollEnd):
            task = evt.header.descr.task
            is_ready = evt.header.descr.is_ready

            try:
                tid = known_tasks[task]
            except KeyError:
                print(f"tokio poll end w/o any begin: task={task}")
                continue

            if phys_thread_state.get(evt.header.thread_id) != tid:
                print(f"tokio poll end w/o begin on this thread: thread={evt.header.thread_id} task={task}")
                continue

            del phys_thread_state[evt.header.thread_id]

            if is_ready:
                try:
                    del known_tasks[task]
                except KeyError:
                    print(f"unknown tokio task: task={task}")
                    continue
                emit_task_finish = True
        elif isinstance(evt.header.descr, EventDescrTokioTaskFinish):
            task = evt.header.descr.task

            try:
                tid = known_tasks[task]
            except KeyError:
                continue

            if phys_thread_state.get(evt.header.thread_id) == tid:
                del phys_thread_state[evt.header.thread_id]

            try:
                del known_tasks[task]
            except KeyError:
                continue

        if tid is None:
            try:
                tid = phys_thread_state[evt.header.thread_id]
                task = known_tasks_rev[tid]
            except KeyError:
                tid = evt.header.thread_id

        if task is None:
            thread_name = f"thread: {evt.header.thread_id}: {evt.header.thread_name}"
        else:
            thread_name = f"tokio task: {task}"

        # emit "thread name" metadata events
        if tid not in thread_names:
            thread_names[tid] = thread_name

            if not process_metadata_done:
                yield Event(
                    header=EventHeader(
                        thread_name=thread_name,
                        thread_id=tid,
                        core=evt.header.core,
                        ts=0,
                        descr=EventDescrProcessName(
                            name="tokio2chrome",
                        ),
                    ),
                    backtrace=None,
                )
                yield Event(
                    header=EventHeader(
                        thread_name=thread_name,
                        thread_id=tid,
                        core=evt.header.core,
                        ts=0,
                        descr=EventDescrProcessSortIndex(
                            sort_index=0,
                        ),
                    ),
                    backtrace=None,
                )
                process_metadata_done = True

            yield Event(
                header=EventHeader(
                    thread_name=thread_name,
                    thread_id=tid,
                    core=evt.header.core,
                    ts=0,
                    descr=EventDescrThreadName(
                        name=thread_name,
                    ),
                ),
                backtrace=None,
            )
            yield Event(
                header=EventHeader(
                    thread_name=thread_name,
                    thread_id=tid,
                    core=evt.header.core,
                    ts=0,
                    descr=EventDescrThreadSortIndex(
                        sort_index=tid,
                    ),
                ),
                backtrace=None,
            )

            if task is not None:
                yield Event(
                    header=EventHeader(
                        thread_name=thread_name,
                        thread_id=tid,
                        core=evt.header.core,
                        ts=evt.header.ts,
                        descr=EventDescrTokioTaskStart(
                            task=task,
                        ),
                    ),
                    backtrace=None,
                )

        # emit actual event
        yield Event(
            header=EventHeader(
                thread_name=thread_name,
                thread_id=tid,
                core=evt.header.core,
                ts=evt.header.ts,
                descr=evt.header.descr,
            ),
            backtrace=evt.backtrace,
        )

        # add task finish event
        if task is not None and emit_task_finish:
            yield Event(
                header=EventHeader(
                    thread_name=thread_name,
                    thread_id=tid,
                    core=evt.header.core,
                    ts=evt.header.ts,
                    descr=EventDescrTokioTaskFinish(
                        task=task,
                    ),
                ),
                backtrace=None,
            )

    for task in sorted(known_tasks):
        print(f"task never ended: task={task}")


def main() -> None:
    # accumulates all events for JSON output
    events = []

    with open("perf.txt") as f_in:
        for evt in process_events(parse(f_in)):
            events.append(evt.to_chrome())

    out = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
    }
    with open("perf.json", "w") as f_out:
        json.dump(out, f_out)


if __name__ == "__main__":
    main()
