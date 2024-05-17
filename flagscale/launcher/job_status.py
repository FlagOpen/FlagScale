from enum import Enum


class JobStatus(Enum):
    RUNNING = "Running"
    TRANSITIONAL = "Transitional (Stopping or Starting)"
    COMPLETED_OR_IDLE = "Completed or Not Started"
