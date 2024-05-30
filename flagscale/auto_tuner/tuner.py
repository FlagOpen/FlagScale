import os
import copy
import time
import logging

from omegaconf import DictConfig, OmegaConf

from flagscale.launcher.runner import SSHRunner
from flagscale.launcher.job_status import JobStatus

from .search import Searcher
from .prune import Pruner
from .generate import Generator
from .record import Recorder


class AutoTuner:

    def __init__(self, config: DictConfig):
        # Set logger
        OmegaConf.set_struct(config, False)
        logger = logging.getLogger("FlagScale-AutoTuner")
        logger.setLevel(logging.INFO)

        dir_path = os.path.join(config.experiment.exp_dir, "auto_tuner")
        log_path = os.path.join(
            dir_path,
            "auto_tuner.log",
        )
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        handler = logging.FileHandler(log_path, mode="w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger

        # Deepcopy the original config to isolate from each task config
        # Modify the orig config when run best task
        self.orig_config = config
        self.config = copy.deepcopy(config)

        # Set config of auto tuner
        if "auto_tuner" not in self.config:
            self.config.auto_tuner = {}

        # Add nodes, nproc_per_node, cards to build search space or prune
        nnodes = config.experiment.runner.nnodes
        nproc_per_node = config.experiment.runner.nproc_per_node
        self.config.auto_tuner.nproc_per_node = nproc_per_node
        self.config.auto_tuner.nnodes = nnodes
        self.config.auto_tuner.cards = nnodes * nproc_per_node
        # TODO: Set platform envs
        if "platform" not in self.config.auto_tuner:
            self.config.auto_tuner.platform = {}

        # Build core sub modules, such as Searcher, Pruner, Generator and Recorder
        self.searcher = Searcher(self.config)
        self.pruner = Pruner(self.config)
        self.generator = Generator(self.config)
        self.recorder = Recorder(self.config)

        # Each task has its own runner
        self.runner = None

        # Set tuner configs
        # The interval of task monitoring
        if "control" not in self.config.auto_tuner:
            self.config.auto_tuner.control = {}
        self.interval = self.config.auto_tuner.control.get("interval", 10)

        # The max time per task, unit: second
        # NOTE: The task will be stopped if the time is reached or done.
        self.max_time_per_task = self.config.auto_tuner.control.get(
            "max_time_per_task", 300)

        # The max time of auto tuner, if None, no limit.
        self.max_time = self.config.auto_tuner.control.get("max_time", None)

        # The start time of each task, used to control each task when stop
        self.start_task_time = None

        # The start time of tuner, used to control the tuner when stop
        self.start_time = time.time()

        # History strategy
        self.history = []

        # Task id
        self.idx = 0

        # Checkout search mode on the platform
        self.has_checkout = False

    def tune(self):
        """
        Tune the model performance, the steps are:
            Step1. Generate the task to run
            Step2. Run task and monitor
            Step3. Record task
            Step4. Loop 1-3 until stop
            Step5. Run the best task
        """
        tuner_start_time = time.time()
        while not self.need_stop():
            self.gen()
            if not self.cur_strategy:
                break
            self.logger.info(f"Run task_{self.idx}: {self.cur_strategy}")
            self.run()
            self.logger.info(f"Monitor task_{self.idx}:")
            self.monitor()
            self.logger.info(f"Record task_{self.idx}:")
            self.record()

            if self.cur_strategy[
                    "performance"] and self.config.auto_tuner.platform.get(
                        "airs_switch", False) and not self.has_checkout:
                self.checkout()

        tuner_end_time = time.time()
        self.logger.info(
            f"AutoTuner Ended in {tuner_end_time - tuner_start_time} seconds.")
        # TODO: Run the best task

    def need_stop(self):
        """Judge whether need to stop tuning."""
        end_time = time.time()
        # If the max time of tuner is reached, stop
        if self.max_time:
            if end_time - self.start_time > self.max_time:
                return True

        # If no task to tune, stop
        if self.searcher.has_done():
            return True

        # TODO: Add task limits to control the tuner

        return False

    def checkout(self, mode="performance"):
        if not self.has_checkout:
            self.searcher.algo.checkout(mode)
            self.has_checkout = True

    def gen(self):
        """Generate a task to run."""
        # 1. Get a strategy from searcher
        # 2. Whether prune by pruner
        # 3. If not pruned, generate the task by generator
        strategy = self.searcher.search()
        while strategy and self.pruner.prune(strategy, self.history):
            strategy = self.searcher.search()
        if strategy:
            self.idx += 1
            strategy["idx"] = self.idx
            self.logger.info(
                f"Searching {self.idx+self.pruner.pruned_count} / {len(self.searcher.strategies)} strategy, Pruned {self.pruner.pruned_count} strategy."
            )
            self.logger.info(f"Generate task_{self.idx}")
            self.cur_strategy = strategy
            self.cur_task = self.generator.gen(strategy)
        else:
            self.cur_strategy = None

    def run(self, task=None):
        # Instantiate a runner and run the task
        if task is None:
            task = self.cur_task
        self.runner = SSHRunner(task)
        self.runner.run()
        # set start time
        self.task_start_time = time.time()

    def monitor(self):
        """Monitor the task until task timeout or completed."""
        # Sleep 3s to ensure the task is started
        time.sleep(3)

        while True:
            # If the task timeout, stop monitoring
            end_time = time.time()
            if self.idx == 1:
                max_time_per_task = 600
            else:
                max_time_per_task = self.max_time_per_task
            if end_time - self.task_start_time > max_time_per_task:
                self.runner.stop()
                self.cur_strategy["stopped_by_tuner"] = True
                break
            # If the task is completed or idle, stop monitoring
            try:
                status = self.runner._query_status()
                self.logger.info(
                    f"task_{self.cur_strategy['idx']} status: {status.name}")
                if status == JobStatus.COMPLETED_OR_IDLE:
                    break
            except Exception as e:
                self.logger.info(e)
                time.sleep(self.interval)
            time.sleep(self.interval)

        end_time = time.time()
        self.cur_strategy["elapsed_time"] = round(
            end_time - self.task_start_time, 2)
        self.logger.info("task_{} monitor time: {:.2f}s".format(
            self.cur_strategy["idx"], self.cur_strategy["elapsed_time"]))

    def record(self):
        """Record the task result to csv"""
        self.recorder.record(self.cur_task, self.cur_strategy)
        self.recorder.save(self.history)
