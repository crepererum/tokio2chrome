#!/usr/bin/env python3
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, Generator, List, Union


PID = 1
VIRT_THREAD_OFFSET = 2**32


class EventDescrType(Enum):
    BEGIN = "B"
    END = "E"
    INSTANT = "i"


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
    ty: EventDescrType = EventDescrType.END


@dataclass
class EventDescOther:
    content: str
    ty: EventDescrType = EventDescrType.INSTANT


EventDescr = Union[
    EventDescrSysEnter,
    EventDescrSysExit,
    EventDescrTokioTaskPollBegin,
    EventDescrTokioTaskPollEnd,
    EventDescOther,
]


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
    backtrace: List[str]

    def to_chrome(self, tid: int) -> Dict[str, Any]:
        out = {
            "pid": PID,
            "tid": tid,
            "ts": self.header.ts,
            "name": str(self.header.descr),
            "cat": str(type(self.header.descr)),
            "stack": self.backtrace,
            "ph": self.header.descr.ty.value,
        }

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
        )
    else:
        return EventDescOther(
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


def metadata_event(name: str, tid: int, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "ph": "M",
        "pid": PID,
        "tid": tid,
        "ts": 0,
        "args": args,
    }


def main() -> None:
    # accumulates all events for JSON output
    events = []

    # ID counter for virtual threads (one for each tokio task)
    virt_thread_counter = 0

    # maps thread ID to virtual thread ID.
    # If a thrad ID is NOT in this dict, use its real ID.
    phys_thread_state: Dict[int, int] = {}

    # maps task ID to virtual thread ID and reverse
    known_tasks: Dict[int, int] = {}
    known_tasks_rev: Dict[int, int] = {}

    # maps threads to names
    thread_names: Dict[int, str] = {}

    # flags if we have set up process-level metadata
    process_metadata_done = False

    with open("perf.txt") as f_in:
        for evt in parse(f_in):
            if isinstance(evt.header.descr, EventDescrTokioTaskPollBegin):
                task = evt.header.descr.task

                if task not in known_tasks:
                    known_tasks[task] = VIRT_THREAD_OFFSET - virt_thread_counter
                    virt_thread_counter += 1

                phys_thread_state[evt.header.thread_id] = known_tasks[task]
                known_tasks_rev[known_tasks[task]] = task
            elif isinstance(evt.header.descr, EventDescrTokioTaskPollEnd):
                task = evt.header.descr.task
                is_ready = evt.header.descr.is_ready

                try:
                    del phys_thread_state[evt.header.thread_id]
                except KeyError:
                    print(f"tokio poll end w/o begin: thread={evt.header.thread_id} task={task}")

                if is_ready:
                    try:
                        del known_tasks[task]
                    except KeyError:
                        print(f"unknown tokio task: task={task}")

            try:
                tid = phys_thread_state[evt.header.thread_id]
                task = known_tasks_rev[tid]
                thread_name = f"tokio task: {task}"
            except KeyError:
                tid = evt.header.thread_id
                thread_name = f"thread: {evt.header.thread_id}: {evt.header.thread_name}"

            # emit "thread name" metadata events
            if tid not in thread_names:
                thread_names[tid] = thread_name

                if not process_metadata_done:
                    events += [
                        metadata_event(
                            "process_name",
                            tid,
                            {
                                "name": "tokio2chrome",
                            },
                        ),
                        metadata_event(
                            "process_sort_index",
                            tid,
                            {
                                "sort_index": 0,
                            },
                        ),
                    ]
                    process_metadata_done = True

                events += [
                    metadata_event(
                        "thread_name",
                        tid,
                        {
                            "name": thread_name,
                        },
                    ),
                    metadata_event(
                        "thread_sort_index",
                        tid,
                        {
                            "sort_index": tid,
                        },
                    ),
                ]

            # emit actual event
            events.append(evt.to_chrome(tid))


    for task in sorted(known_tasks):
        print(f"task never ended: task={task}")


    out = {
        "traceEvents": events,
        "displayTimeUnit": "s",
    }
    with open("perf.json", "w") as f_out:
        json.dump(out, f_out)


if __name__ == "__main__":
    main()
