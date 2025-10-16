from collections import deque, defaultdict
import itertools
import time
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Events (yielded by tasks)
# -------------------------
class SysCall: pass

class Sleep(SysCall):
    def __init__(self, ticks): self.ticks = ticks

class Yield(SysCall):
    pass

class Wait(SysCall):
    def __init__(self, sync): self.sync = sync

class Signal(SysCall):
    def __init__(self, sync): self.sync = sync

class Terminate(SysCall):
    pass

# -------------------------
# Task and primitives
# -------------------------
TASK_STATES = ("READY", "RUNNING", "BLOCKED", "SLEEPING", "TERMINATED")

class Task:
    _ids = itertools.count(1)
    def __init__(self, gen, name=None, priority=1, quantum=2):
        self.tid = next(Task._ids)
        self.name = name or f"task{self.tid}"
        self.gen = gen
        self.priority = priority
        self.quantum = quantum
        self.state = "READY"
        self.wake_tick = None
        self.waiting_on = None

    def __repr__(self):
        return f"<{self.name}#{self.tid} p={self.priority} s={self.state}>"

# -------------------------
# Sync primitives
# -------------------------
class Mutex:
    def __init__(self, name=None):
        self.owner = None
        self.waiters = deque()
        self.name = name or "mutex"

    def __repr__(self):
        return f"<Mutex {self.name} owner={self.owner}>"

class Semaphore:
    def __init__(self, initial=1, name=None):
        self.count = initial
        self.waiters = deque()
        self.name = name or "sem"

    def __repr__(self):
        return f"<Semaphore {self.name} count={self.count}>"

# -------------------------
# Scheduler (with timeline)
# -------------------------
class Scheduler:
    def __init__(self, tick_hz=1000, log=True):
        self.tick = 0
        self.log_enabled = log
        self.ready = defaultdict(deque)
        self.tasks = {}
        self.sleeping = []
        self.blocked = set()
        self.tick_hz = tick_hz
        # timeline: list of (tick, task_name)
        self.timeline = []
        self.idle_task = self.create_task(self.idle(), name="idle", priority=0, quantum=1)

    def log(self, *args):
        if self.log_enabled:
            print(f"[{self.tick:05d}] ", *args)

    def create_task(self, gen, name=None, priority=1, quantum=2):
        task = Task(gen, name=name, priority=priority, quantum=quantum)
        self.tasks[task.tid] = task
        self.ready[priority].append(task)
        self.log("Created", task)
        return task

    def idle(self):
        while True:
            yield Sleep(1)

    def pick_next_task(self):
        for p in sorted(self.ready.keys(), reverse=True):
            q = self.ready[p]
            while q:
                t = q.popleft()
                if t.state == "READY":
                    return t
        return None

    def tick_once(self):
        self.tick += 1

        # Wake sleeping tasks
        to_wake = [t for t in self.tasks.values() if t.state == "SLEEPING" and t.wake_tick is not None and t.wake_tick <= self.tick]
        for t in to_wake:
            t.state = "READY"
            t.wake_tick = None
            self.ready[t.priority].append(t)
            self.log("Wake", t)

        task = self.pick_next_task()
        if not task:
            self.timeline.append((self.tick, "idle"))
            return
        self.timeline.append((self.tick, task.name))

        task.state = "RUNNING"
        remaining = task.quantum
        self.log("Run", task, f"quantum={remaining}")
        while remaining > 0:
            try:
                ev = next(task.gen)
            except StopIteration:
                task.state = "TERMINATED"
                self.log("Terminated", task)
                break

            if isinstance(ev, Sleep):
                task.state = "SLEEPING"
                task.wake_tick = self.tick + ev.ticks
                self.log(task, "sleep for", ev.ticks, "-> wake at", task.wake_tick)
                break
            elif isinstance(ev, Yield) or ev is None:
                task.state = "READY"
                self.ready[task.priority].append(task)
                self.log(task, "yield")
                break
            elif isinstance(ev, Wait):
                sync = ev.sync
                if isinstance(sync, Mutex):
                    if sync.owner is None:
                        sync.owner = task
                        self.log(task, "acquired", sync)
                        pass
                    else:
                        task.state = "BLOCKED"
                        task.waiting_on = sync
                        sync.waiters.append(task)
                        self.blocked.add(task)
                        self.log(task, "blocked waiting for", sync)
                        break
                elif isinstance(sync, Semaphore):
                    if sync.count > 0:
                        sync.count -= 1
                        self.log(task, "sem got, new count", sync.count)
                    else:
                        task.state = "BLOCKED"
                        task.waiting_on = sync
                        sync.waiters.append(task)
                        self.blocked.add(task)
                        self.log(task, "blocked waiting for sem", sync)
                        break
                else:
                    raise RuntimeError("Unknown sync primitive")
            elif isinstance(ev, Signal):
                sync = ev.sync
                if isinstance(sync, Mutex):
                    if sync.owner is not None and sync.owner == task:
                        sync.owner = None
                        self.log(task, "released", sync)
                        if sync.waiters:
                            nxt = sync.waiters.popleft()
                            sync.owner = nxt
                            if nxt.state == "BLOCKED":
                                nxt.state = "READY"
                                nxt.waiting_on = None
                                self.ready[nxt.priority].append(nxt)
                                self.blocked.discard(nxt)
                                self.log("woken", nxt, "now owner of", sync)
                    else:
                        self.log("Warning:", task, "tried to release mutex it doesn't own", sync)
                elif isinstance(sync, Semaphore):
                    sync.count += 1
                    self.log(task, "sem signal -> count", sync.count)
                    if sync.waiters:
                        nxt = sync.waiters.popleft()
                        if nxt.state == "BLOCKED":
                            nxt.state = "READY"
                            nxt.waiting_on = None
                            self.ready[nxt.priority].append(nxt)
                            self.blocked.discard(nxt)
                            self.log("woken", nxt, "by sem", sync)
                else:
                    raise RuntimeError("Unknown sync primitive")
            elif isinstance(ev, Terminate):
                task.state = "TERMINATED"
                self.log("TERMINATED", task)
                break
            else:
                # Unknown: treat as voluntary yield
                task.state = "READY"
                self.ready[task.priority].append(task)
                self.log(task, "yield (unknown event)", ev)
                break
            remaining -= 1

        if task.state == "RUNNING":
            task.state = "READY"
            self.ready[task.priority].append(task)
            self.log(task, "quantum expired -> preempted")

    def run(self, max_ticks=1000, realtime=False, realtime_delay=0.0):
        for _ in range(max_ticks):
            self.tick_once()
            if realtime and realtime_delay:
                time.sleep(realtime_delay)

    # -------------------------
    # Timeline helpers
    # -------------------------
    def get_timeline_df(self):
        if not self.timeline:
            return pd.DataFrame(columns=["tick", "task"])
        df = pd.DataFrame(self.timeline, columns=["tick", "task"])
        return df

    def save_timeline_csv(self, filename="rtos_timeline.csv"):
        df = self.get_timeline_df()
        df.to_csv(filename, index=False)
        if self.log_enabled:
            self.log("Saved timeline to", filename)

    def plot_gantt(self, df=None, title="RTOS Gantt Timeline"):
        if df is None:
            df = self.get_timeline_df()
        if df.empty:
            print("No timeline data to plot.")
            return

        intervals = []
        cur_task = None
        cur_start = None
        last_tick = None
        for _, row in df.iterrows():
            tick = int(row["tick"])
            task = row["task"]
            if cur_task is None:
                cur_task = task
                cur_start = tick
            elif task == cur_task and tick == last_tick + 1:
                pass
            else:
                intervals.append((cur_task, cur_start, last_tick))
                cur_task = task
                cur_start = tick
            last_tick = tick
        # add last
        intervals.append((cur_task, cur_start, last_tick))

        iv_df = pd.DataFrame(intervals, columns=["task", "start", "end"])
        iv_df["duration"] = iv_df["end"] - iv_df["start"] + 1

        tasks = iv_df["task"].unique().tolist()
        y_positions = {task: i for i, task in enumerate(tasks[::-1])}

        fig, ax = plt.subplots(figsize=(12, 1 + len(tasks) * 0.5))
        for _, r in iv_df.iterrows():
            y = y_positions[r["task"]]
            ax.broken_barh([(r["start"], r["duration"])], (y - 0.4, 0.8))

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        ax.set_xlabel("Tick (time unit)")
        ax.set_title(title)
        ax.set_xlim(df["tick"].min() - 1, df["tick"].max() + 1)
        ax.set_ylim(-1, len(tasks))
        ax.grid(True, axis="x", linestyle=":")
        plt.tight_layout()
        plt.show()

# -------------------------
# Helper decorators for tasks
# -------------------------
def rtos_task(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# -------------------------
# Example Tasks
# -------------------------
@rtos_task
def blink(name="LED", on_ticks=2, off_ticks=3, cycles=5):
    for i in range(cycles):
        print(f"    {name}: ON  (cycle {i+1})")
        yield Sleep(on_ticks)
        print(f"    {name}: OFF (cycle {i+1})")
        yield Sleep(off_ticks)
    print(f"    {name}: DONE")
    yield Terminate()

@rtos_task
def cpu_heavy(name="CPU", work_ticks=1, loops=10):
    for i in range(loops):
        print(f"    {name}: working {i+1}/{loops}")
        yield Yield()
    print(f"    {name}: DONE")
    yield Terminate()

@rtos_task
def producer(sem, count=5, name="prod"):
    for i in range(count):
        print(f"    {name}: producing {i}")
        yield Sleep(1)
        yield Signal(sem)
    print(f"    {name}: DONE")
    yield Terminate()

@rtos_task
def consumer(sem, name="cons"):
    while True:
        print(f"    {name}: waiting for item")
        yield Wait(sem)
        print(f"    {name}: got item, consuming")
        yield Sleep(2)
    yield Terminate()

@rtos_task
def mutex_user(m, name="MUser"):
    print(f"    {name}: trying to lock")
    yield Wait(m)
    print(f"    {name}: acquired, doing work")
    yield Sleep(3)
    print(f"    {name}: releasing")
    yield Signal(m)
    yield Terminate()

# -------------------------
# Demo / main
# -------------------------
def main():
    sched = Scheduler(log=True)

    sched.create_task(blink("LED1", on_ticks=2, off_ticks=3, cycles=6), name="blink1", priority=2, quantum=1)
    sched.create_task(cpu_heavy(name="CPU1", loops=10), name="cpu1", priority=1, quantum=2)

    sem = Semaphore(initial=0, name="item_sem")
    sched.create_task(producer(sem, count=8, name="producer"), name="producer", priority=2, quantum=1)
    sched.create_task(consumer(sem, name="consumer"), name="consumer", priority=1, quantum=1)

    m = Mutex("mA")
    sched.create_task(mutex_user(m, name="muA"), name="muA", priority=3, quantum=1)
    sched.create_task(mutex_user(m, name="muB"), name="muB", priority=2, quantum=1)

    max_ticks = 80
    sched.run(max_ticks=max_ticks)

    df = sched.get_timeline_df()
    print("\nTimeline (tick -> task) sample:")
    print(df.head(60).to_string(index=False))

    sched.save_timeline_csv("rtos_timeline.csv")
    sched.plot_gantt(df, title=f"RTOS Gantt (first {max_ticks} ticks)")

if __name__ == "__main__":
    main()
