"""
This module provides utilities for managing shared counters in a multiprocessing
environment. It includes functionality to allocate unique positions for progress
indicators (such as `tqdm` progress bars) in a thread-safe manner.

Configuration:
    USE_MULTIPROCESSING (bool): Set to True if multiprocessing is desired for the 
        application. Default is False, which means single-threaded operation.

Constants:
    START_POSITION (int): The starting index for position allocation.
    NUM_PROCESSES (int): The number of processes to be supported.
    MAX_POSITION (int): The maximum position index, calculated as the sum of 
        START_POSITION and NUM_PROCESSES.

Shared Resources:
    PROCESS_IDX (multiprocessing.Value): A synchronized integer value to track 
        the current position index.
    COUNTER_LOCK (multiprocessing.Lock): A lock to ensure thread-safe access to 
        shared resources.
    USED_INDICES (multiprocessing.Array): An array to track which positions are 
        currently in use.

Functions:
    increment_shared_counter() -> int:
        Allocates a unique position for progress bars, ensuring no overlap.
    
    release_shared_counter(tqdm_position: int):
        Releases a previously allocated position, making it available for reuse.
"""

from multiprocessing import Value, Lock, Array
from multiprocessing.sharedctypes import SynchronizedBase, SynchronizedArray
from multiprocessing.synchronize import Lock as MPLock
from ctypes import c_int


# Configuration for multiprocessing usage
USE_MULTIPROCESSING: bool = True  # Set to True if multiprocessing is desired

# Constants for position management
START_POSITION: int = 5
NUM_PROCESSES: int = 10
MAX_POSITION: int = NUM_PROCESSES + START_POSITION

# Shared resources for position tracking
PROCESS_IDX: SynchronizedBase = Value(c_int, START_POSITION, lock=True)
COUNTER_LOCK: MPLock = Lock()

# Initialize an array with MAX_POSITION zeros to track used positions
USED_INDICES: SynchronizedArray = Array(c_int, [0] * MAX_POSITION)

def increment_shared_counter() -> int:
    """
    Increment the shared counter to allocate a unique position for tqdm progress bars.

    Returns:
        int: The allocated position index.

    Raises:
        RuntimeError: If no available position can be allocated.
    """
    if USE_MULTIPROCESSING:
        with COUNTER_LOCK:
            for _ in range(MAX_POSITION - START_POSITION):
                current_position: int = PROCESS_IDX.value
                if not USED_INDICES[current_position]:
                    USED_INDICES[current_position] = c_int(1)  # Mark as used
                    # Update the shared value
                    PROCESS_IDX.value = (current_position + 1) % MAX_POSITION
                    return current_position

                # Move to the next position
                PROCESS_IDX.value = (current_position + 1) % MAX_POSITION

            # If no position is available, raise an error
            raise RuntimeError("No available positions in USED_INDICES")
    else:
        return START_POSITION

def release_shared_counter(tqdm_position: int):
    """
    Release a previously allocated position.

    Args:
        tqdm_position (int): The position to be released.
    """
    if USE_MULTIPROCESSING:
        with COUNTER_LOCK:
            USED_INDICES[tqdm_position] = c_int(0)
