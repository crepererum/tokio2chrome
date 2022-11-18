#!/usr/bin/env python3
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import json
from typing import Generator, List, Union


class EventDescrType(Enum):
    BEGIN = 1
    END = 2
    INSTANT = 3


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

    def to_chrome(self, tid: int):
        out = {
            "pid": 1,
            "tid": tid,
            "ts": self.header.ts,
            "name": str(self.header.descr),
            "cat": str(type(self.header.descr)),
            "stack": self.backtrace,
        }

        if self.header.descr.ty == EventDescrType.BEGIN:
            out["ph"] = "B"
        elif self.header.descr.ty == EventDescrType.END:
            out["ph"] = "E"
        elif self.header.descr.ty == EventDescrType.INSTANT:
            out["ph"] = "i"
            out["s"] = "t"
        else:
            raise Exception(f"Unknown event description type: {self.descr.ty}")

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
    header_head, ts = header_head.rsplit(" ", maxsplit=1)
    ts = float(ts)

    header_head, core = header_head.rsplit(" ", maxsplit=1)
    assert core.startswith("[") and core.endswith("]")
    core = int(core[1:-1])

    header_head, thread_id = header_head.rsplit(" ", maxsplit=1)
    thread_id = int(thread_id)

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
    line = None

    while True:
        if line is None:
            try:
                line = next(lines)
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
                line = next(lines)
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


VIRT_THREAD_OFFSET = 2**32


def main() -> None:
    events = []
    virt_thread_counter = 0
    phys_thread_state = {}
    known_tasks = {}

    with open("perf.txt") as f_in:
        for evt in parse(f_in):
            if isinstance(evt.header.descr, EventDescrTokioTaskPollBegin):
                task = evt.header.descr.task

                if task not in known_tasks:
                    known_tasks[task] = VIRT_THREAD_OFFSET - virt_thread_counter
                    virt_thread_counter += 1

                phys_thread_state[evt.header.thread_id] = known_tasks[task]
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
            except KeyError:
                tid = evt.header.thread_id

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