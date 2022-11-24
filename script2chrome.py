#!/usr/bin/env python3
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
import json
import re
from typing import cast, Generator, Tuple, TypeAlias
from weakref import ref, ReferenceType


PID = 1
VIRT_THREAD_OFFSET = 2**32


# See https://github.com/python/typing/issues/182#issuecomment-1320974824
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

SNAKE_CASE_RE = re.compile(r"(?<!^)(?=[A-Z])")

# See https://stackoverflow.com/a/1176023
def camel2snake(s: str) -> str:
    return SNAKE_CASE_RE.sub("_", s).lower()


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
class EventDescrTokioTaskBlockingBegin:
    ty: EventDescrType = EventDescrType.BEGIN


@dataclass
class EventDescrTokioTaskBlockingEnd:
    ty: EventDescrType = EventDescrType.END


@dataclass
class EventDescrStackSample:
    ty: EventDescrType = EventDescrType.INSTANT


@dataclass
class EventDescrStackEnter:
    name: str
    ty: EventDescrType = EventDescrType.BEGIN


@dataclass
class EventDescrStackExit:
    name: str
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
class EventDescrCounter:
    content: str
    ty: EventDescrType = EventDescrType.INSTANT


@dataclass
class EventDescrOther:
    content: str
    ty: EventDescrType = EventDescrType.INSTANT


EventDescr = (
    EventDescrSysEnter
    | EventDescrSysExit
    | EventDescrTokioTaskBlockingBegin
    | EventDescrTokioTaskBlockingEnd
    | EventDescrTokioTaskStart
    | EventDescrTokioTaskFinish
    | EventDescrTokioTaskPollBegin
    | EventDescrTokioTaskPollEnd
    | EventDescrProcessName
    | EventDescrProcessSortIndex
    | EventDescrThreadName
    | EventDescrThreadSortIndex
    | EventDescrStackSample
    | EventDescrStackEnter
    | EventDescrStackExit
    | EventDescrCounter
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

    @property
    def category(self) -> str:
        return camel2snake(
            type(self.header.descr)
            .__name__.removeprefix("EventDescr")
            .removesuffix("Enter")
            .removesuffix("Exit")
            .removesuffix("Begin")
            .removesuffix("End")
            .removesuffix("Start")
            .removesuffix("Finish")
        )

    @property
    def name(self) -> str:
        if isinstance(self.header.descr, (EventDescrStackEnter, EventDescrStackExit)):
            return self.header.descr.name
        else:
            return self.category

    def to_chrome(self) -> JSON:
        name = self.name
        category = self.category

        out: dict[str, JSON] = {
            "pid": PID,
            "tid": self.header.thread_id,
            # ts is in micros
            "ts": self.header.ts * 1_000_000,
            "cat": category,
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

        # we emit the stack as events, so do NOT duplicate the data
        #   if self.backtrace is not None:
        #       out["stack"] = cast(JSON, self.backtrace)

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
    elif "sdt_tokio:task_blocking_begin" in s:
        return EventDescrTokioTaskBlockingBegin()
    elif "sdt_tokio:task_blocking_end" in s:
        return EventDescrTokioTaskBlockingEnd()
    elif "cycles/" in s:
        return EventDescrCounter(content=s)
    elif "instructions/" in s:
        return EventDescrCounter(content=s)
    else:
        return EventDescrOther(content=s)


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


def parse_perf_script_output(lines: Iterable[str]) -> Generator[Event, None, None]:
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


def recover_virtual_threads(events: Iterable[Event]) -> Generator[Event, None, None]:
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

    # remembers stashed phys_thread_state entries (due to bocking tasks)
    phys_thread_state_stash: dict[int, list[int | None]] = {}

    for evt in events:
        task = None
        tid = None
        emit_task_finish = False

        if isinstance(evt.header.descr, EventDescrTokioTaskPollBegin):
            task = evt.header.descr.task

            if evt.header.thread_id in phys_thread_state:
                old_tid = phys_thread_state[evt.header.thread_id]
                old_task = known_tasks_rev[old_tid]
                print(
                    f"already have a task running on this thread: thread={evt.header.thread_id} new_task={task} old_task={old_task}"
                )
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
                print(
                    f"tokio poll end w/o begin on this thread: thread={evt.header.thread_id} task={task}"
                )
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
        elif isinstance(evt.header.descr, EventDescrTokioTaskBlockingBegin):
            tid = phys_thread_state.get(evt.header.thread_id)
            if tid is not None:
                task = known_tasks_rev[tid]
                del phys_thread_state[evt.header.thread_id]
            else:
                print(
                    f"No tokio task active, but starting blocking task: thread={evt.header.thread_id}"
                )

            if evt.header.thread_id not in phys_thread_state_stash:
                phys_thread_state_stash[evt.header.thread_id] = []
            phys_thread_state_stash[evt.header.thread_id].append(tid)
        elif isinstance(evt.header.descr, EventDescrTokioTaskBlockingEnd):
            try:
                tid = phys_thread_state_stash[evt.header.thread_id].pop(-1)
            except (KeyError, IndexError):
                print(
                    f"No stashed tokio task, but finished blocking task: thread={evt.header.thread_id}"
                )
                continue

            if tid is not None:
                task = known_tasks_rev[tid]
                phys_thread_state[evt.header.thread_id] = tid
            else:
                try:
                    del phys_thread_state_stash[evt.header.thread_id]
                except KeyError:
                    pass

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

        if tid not in thread_names:
            thread_names[tid] = thread_name
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


def add_metadata_events(events: Iterable[Event]) -> Generator[Event, None, None]:
    # flags if we have set up process-level metadata
    process_metadata_done = False

    # flags if we have set up thread-level metadata
    thread_metadata_done: set[int] = set()

    for evt in events:
        if not process_metadata_done:
            yield Event(
                header=EventHeader(
                    thread_name=evt.header.thread_name,
                    thread_id=evt.header.thread_id,
                    core=evt.header.core,
                    ts=evt.header.ts,
                    descr=EventDescrProcessName(
                        name="tokio2chrome",
                    ),
                ),
                backtrace=None,
            )
            yield Event(
                header=EventHeader(
                    thread_name=evt.header.thread_name,
                    thread_id=evt.header.thread_id,
                    core=evt.header.core,
                    ts=evt.header.ts,
                    descr=EventDescrProcessSortIndex(
                        sort_index=0,
                    ),
                ),
                backtrace=None,
            )
            process_metadata_done = True

        # emit "thread name" metadata events
        if evt.header.thread_id not in thread_metadata_done:
            yield Event(
                header=EventHeader(
                    thread_name=evt.header.thread_name,
                    thread_id=evt.header.thread_id,
                    core=evt.header.core,
                    ts=evt.header.ts,
                    descr=EventDescrThreadName(
                        name=evt.header.thread_name,
                    ),
                ),
                backtrace=None,
            )
            yield Event(
                header=EventHeader(
                    thread_name=evt.header.thread_name,
                    thread_id=evt.header.thread_id,
                    core=evt.header.core,
                    ts=evt.header.ts,
                    descr=EventDescrThreadSortIndex(
                        sort_index=evt.header.thread_id,
                    ),
                ),
                backtrace=None,
            )
            thread_metadata_done.add(evt.header.thread_id)

        yield evt


def check_timestamps(events: Iterable[Event]) -> Generator[Event, None, None]:
    last_ts: dict[int, Tuple[float, int]] = {}

    for evt in events:
        try:
            last, seen = last_ts[evt.header.thread_id]
        except KeyError:
            last = 0.0
            seen = 0

        assert evt.header.ts >= last

        yield Event(
            header=EventHeader(
                thread_id=evt.header.thread_id,
                thread_name=evt.header.thread_name,
                core=evt.header.core,
                # Speedscope is a bit weird and may mess up stacks if we have identical timestamps, so shift them a bit
                ts=evt.header.ts + seen / 1_000_000_000,
                descr=evt.header.descr,
            ),
            backtrace=evt.backtrace,
        )

        seen += 1

        last_ts[evt.header.thread_id] = (evt.header.ts, seen)


def fix_thread_start_syscalls(events: Iterable[Event]) -> Generator[Event, None, None]:
    active_threads: set[int] = set()

    for evt in events:
        # threads may start with execve (NR=59) or clone3 (NR=435)
        if (
            (evt.header.thread_id not in active_threads)
            and isinstance(evt.header.descr, EventDescrSysExit)
            and (evt.header.descr.nr in (59, 435))
        ):
            evt = Event(
                header=EventHeader(
                    thread_name=evt.header.thread_name,
                    thread_id=evt.header.thread_id,
                    core=evt.header.core,
                    ts=evt.header.ts,
                    descr=EventDescrOther(
                        content=f"thread enter (NR={evt.header.descr.nr})"
                    ),
                ),
                backtrace=evt.backtrace,
            )

        # ignore metadata and counter events because they may appear while the syscall is still running
        if (evt.header.descr.ty != EventDescrType.METADATA) and (
            not isinstance(evt.header.descr, EventDescrCounter)
        ):
            active_threads.add(evt.header.thread_id)

        yield evt


def is_unknown_frame(s: str) -> bool:
    return "ffffffffffffffff" in s


def is_unknown_trace(backtrace: list[str]) -> bool:
    return (backtrace is not None) and is_unknown_frame(backtrace[-1])


def common_base(bt1: list[str], bt2: list[str]) -> list[str]:
    rev = []
    for i, (f1, f2) in enumerate(zip(reversed(bt1), reversed(bt2))):
        if f1 != f2:
            break
        rev.append(f1)

    return list(reversed(rev))


@dataclass
class Block:
    parent: ReferenceType[Block] | None
    sub: list[Event | Block]

    def calc_base(self) -> Tuple[Block, list[str]]:
        # collect sub backtraces
        base = None
        sub: list[Event | Block] = []

        for x in self.sub:
            backtrace = None
            if isinstance(x, Event):
                # skip events for now that we cannot process
                if (x.backtrace is not None) and not is_unknown_trace(x.backtrace):
                    backtrace = x.backtrace
            else:
                assert isinstance(x, Block)
                x, backtrace = x.calc_base()

            sub.append(x)

            if backtrace is not None:
                if base is None:
                    base = backtrace
                else:
                    base = common_base(base, backtrace)

        base = base or []

        # fill in missing traces
        sub2 = []
        for x in sub:
            if isinstance(x, Event):
                if x.backtrace is None:
                    x = Event(
                        header=x.header,
                        backtrace=base,
                    )
                elif is_unknown_trace(x.backtrace):
                    x = Event(
                        header=x.header,
                        backtrace=x.backtrace + base,
                    )
            sub2.append(x)

        # fix BEGIN/END events
        if sub2:
            begin = sub2[0]
            if isinstance(begin, Event) and (
                begin.header.descr.ty == EventDescrType.BEGIN
            ):
                assert begin.backtrace is not None
                if base != begin.backtrace:
                    head: list[Event | Block] = [
                        Event(
                            header=begin.header,
                            backtrace=base,
                        ),
                        Event(
                            header=EventHeader(
                                thread_name=begin.header.thread_name,
                                thread_id=begin.header.thread_id,
                                core=begin.header.core,
                                ts=begin.header.ts,
                                descr=EventDescrStackSample(),
                            ),
                            backtrace=begin.backtrace,
                        ),
                    ]
                    sub2 = head + sub2[1:]

            end = sub2[-1]
            if isinstance(end, Event) and (end.header.descr.ty == EventDescrType.END):
                assert end.backtrace is not None
                if base != end.backtrace:
                    tail: list[Event | Block] = [
                        Event(
                            header=EventHeader(
                                thread_name=end.header.thread_name,
                                thread_id=end.header.thread_id,
                                core=end.header.core,
                                ts=end.header.ts,
                                descr=EventDescrStackSample(),
                            ),
                            backtrace=end.backtrace,
                        ),
                        Event(
                            header=end.header,
                            backtrace=base,
                        ),
                    ]
                    sub2 = sub2[:-1] + tail

        this = Block(
            parent=self.parent,
            sub=sub2,
        )
        return this, base

    def flatten(self) -> Generator[Event, None, None]:
        for x in self.sub:
            if isinstance(x, Event):
                yield x
            else:
                yield from x.flatten()


def fix_stack_blocks(events: Iterable[Event]) -> Generator[Event, None, None]:
    thread_roots: dict[int, Block] = {}
    threads: dict[int, Block] = {}

    for evt in events:
        try:
            current_block = threads[evt.header.thread_id]
        except KeyError:
            current_block = Block(parent=None, sub=[])
            thread_roots[evt.header.thread_id] = current_block

        if evt.header.descr.ty == EventDescrType.BEGIN:
            sub_block = Block(parent=ref(current_block), sub=[evt])
            current_block.sub.append(sub_block)
            current_block = sub_block
        elif evt.header.descr.ty == EventDescrType.END:
            parent = current_block.parent

            if (
                (parent is None)
                or (len(current_block.sub) < 1)
                or (not isinstance(current_block.sub[0], Event))
                or (current_block.sub[0].name != evt.name)
            ):
                print(f"end event w/o begin: thread={evt.header.thread_id} evt={evt}")
                evt = Event(
                    header=EventHeader(
                        thread_name=evt.header.thread_name,
                        thread_id=evt.header.thread_id,
                        core=evt.header.core,
                        ts=evt.header.ts,
                        descr=EventDescrOther(
                            content=f"broken END event: {evt.header.descr}"
                        ),
                    ),
                    backtrace=evt.backtrace,
                )
                current_block.sub.append(evt)
            else:
                current_block.sub.append(evt)
                maybe_parent = parent()
                assert maybe_parent is not None, "ref error"
                current_block = maybe_parent
        else:
            current_block.sub.append(evt)

        threads[evt.header.thread_id] = current_block

    for tid in sorted(thread_roots.keys()):
        block = thread_roots[tid]
        block, _ = block.calc_base()
        yield from block.flatten()


@dataclass
class StackState:
    thread_name: str
    core: int
    ts: float
    backtrace: list[str]


def format_stack_frame(idx: int, thread_id: int, perf_output: str) -> str:
    return f"{perf_output} (idx={idx}, tid={thread_id})"


def create_stack_frame_ranges(events: Iterable[Event]) -> Generator[Event, None, None]:
    thread_state: dict[int, StackState] = {}

    for evt in events:
        try:
            state = thread_state[evt.header.thread_id]
        except KeyError:
            state = StackState(
                thread_name=evt.header.thread_name,
                core=evt.header.core,
                ts=evt.header.ts,
                backtrace=[],
            )

        assert evt.backtrace is not None
        base = common_base(evt.backtrace, state.backtrace)
        for i in range(len(state.backtrace) - len(base)):
            yield Event(
                header=EventHeader(
                    thread_id=evt.header.thread_id,
                    thread_name=state.thread_name,
                    core=state.core,
                    ts=state.ts,
                    descr=EventDescrStackExit(
                        name=format_stack_frame(
                            len(state.backtrace) - i,
                            evt.header.thread_id,
                            state.backtrace[i],
                        ),
                    ),
                ),
                backtrace=state.backtrace[i:],
            )
        for i in reversed(range(len(evt.backtrace) - len(base))):
            yield Event(
                header=EventHeader(
                    thread_id=evt.header.thread_id,
                    thread_name=evt.header.thread_name,
                    core=evt.header.core,
                    ts=evt.header.ts,
                    descr=EventDescrStackEnter(
                        name=format_stack_frame(
                            len(evt.backtrace) - i,
                            evt.header.thread_id,
                            evt.backtrace[i],
                        ),
                    ),
                ),
                backtrace=evt.backtrace[i:],
            )

        yield evt

        thread_state[evt.header.thread_id] = StackState(
            thread_name=evt.header.thread_name,
            core=evt.header.core,
            ts=evt.header.ts,
            backtrace=evt.backtrace,
        )

    for tid in sorted(thread_state.keys()):
        state = thread_state[tid]

        for i in range(len(state.backtrace)):
            yield Event(
                header=EventHeader(
                    thread_id=tid,
                    thread_name=state.thread_name,
                    core=state.core,
                    ts=state.ts,
                    descr=EventDescrStackExit(
                        name=format_stack_frame(
                            len(state.backtrace) - i, tid, state.backtrace[i]
                        ),
                    ),
                ),
                backtrace=state.backtrace[i:],
            )


def check_begin_ends(events: Iterable[Event]) -> Generator[Event, None, None]:
    tid_state: dict[int, int] = {}

    for evt in events:
        try:
            idx = tid_state[evt.header.thread_id]
        except KeyError:
            idx = 0

        if evt.header.descr.ty == EventDescrType.BEGIN:
            idx += 1
        elif evt.header.descr.ty == EventDescrType.END:
            idx -= 1

        assert idx >= 0

        tid_state[evt.header.thread_id] = idx
        yield evt


def main() -> None:
    # accumulates all events for JSON output
    events: list[JSON] = []

    with open("perf.txt") as f_in:
        events_it = parse_perf_script_output(f_in)
        events_it = recover_virtual_threads(events_it)
        events_it = add_metadata_events(events_it)
        events_it = fix_thread_start_syscalls(events_it)
        events_it = fix_stack_blocks(events_it)
        events_it = create_stack_frame_ranges(events_it)
        events_it = check_timestamps(events_it)
        events_it = check_begin_ends(events_it)
        for x in events_it:
            events.append(x.to_chrome())

    out = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
    }
    with open("perf.json", "w") as f_out:
        json.dump(out, f_out)


if __name__ == "__main__":
    main()
