# RTOS Scheduler Simulator (Python)


Educational RTOS scheduler simulator that demonstrates preemptive scheduling, priorities, mutex, semaphore, and multicore behavior. Includes Gantt-chart visualization for analysis.


## Quick start


1. Create a virtualenv and activate it (recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Run single-core simulator: `python src/rtos_with_gantt.py`
4. Run multicore simulator: `python src/rtos_multi_core.py`


Outputs: CSV timeline files (rtos_timeline*.csv) and interactive Gantt charts (matplotlib).


## Features:

✅ Round-robin & priority scheduling

✅ Task preemption with quantum (time slice)

✅ Sleep/yield simulation

✅ Semaphore (producer-consumer demo)

✅ Mutex (critical section protection)

✅ Dual-core scheduling (basic SMP behavior)

✅ Gantt-chart visualization of CPU timeline

✅ Clean and modular OOP project in Python
