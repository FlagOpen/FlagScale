import copy
import datetime
import logging
import os
import time

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.auto_tuner.generate import Generator, ServeGenerator
from flagscale.runner.auto_tuner.platform import set_jiuding_platform_args
from flagscale.runner.auto_tuner.prune.pruner import Pruner
from flagscale.runner.auto_tuner.record.recorder import Recorder, ServeRecorder
from flagscale.runner.auto_tuner.search.searcher import Searcher, ServeSearcher
from flagscale.runner.runner_base import JobStatus
from flagscale.runner.runner_serve import SSHServeRunner
from flagscale.runner.runner_train import SSHTrainRunner


class AutoTuner:

    def __init__(self, config: DictConfig):
        # Set logger
        OmegaConf.set_struct(config, False)
        logger = logging.getLogger("FlagScale-AutoTuner")
        logger.setLevel(logging.INFO)

        dir_path = os.path.join(config.experiment.exp_dir, "auto_tuner")
        log_path = os.path.join(dir_path, "tuner.log")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        handler = logging.FileHandler(log_path, mode="w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger

        # Deepcopy the original config to isolate from each task config
        # Modify the orig config when run best task
        self.orig_config = config
        self.config = copy.deepcopy(config)

        # Set config of auto tuner
        if "auto_tuner" not in self.config.experiment:
            self.config.experiment.auto_tuner = {}

        # Add nodes, nproc_per_node, cards to build search space or prune
        assert "experiment" in config, "experiment is not in yaml file."
        assert "runner" in config.experiment, "runner is not in yaml file."
        nnodes = config.experiment.runner.get("nnodes", 1)
        nproc_per_node = config.experiment.runner.get("nproc_per_node", 8)
        self.config.experiment.auto_tuner.nproc_per_node = nproc_per_node
        self.config.experiment.auto_tuner.nnodes = nnodes

        # Set tuner configs
        # The interval of task monitoring
        if "control" not in self.config.experiment.auto_tuner:
            self.config.experiment.auto_tuner.control = {}
        self.interval = self.config.experiment.auto_tuner.control.get("interval", 10)

        # Set platform envs
        if "platform" not in self.config.experiment.auto_tuner:
            self.config.experiment.auto_tuner.platform = {}

        # As long as AIRS_SWITCH has value it means running on the jiuding platform
        if os.environ.get("AIRS_SWITCH", None):
            set_jiuding_platform_args(self.config, self.orig_config)
        # If on other platform, set the platform envs

        self.config.experiment.auto_tuner.cards = (
            self.config.experiment.auto_tuner.nnodes
            * self.config.experiment.auto_tuner.nproc_per_node
        )

        # Build core sub modules, such as Searcher, Pruner, Generator and Recorder
        self.searcher = Searcher(self.config)
        self.pruner = Pruner(self.config)
        self.generator = Generator(self.config)
        self.recorder = Recorder(self.config)

        # Each task has its own runner
        self.runner = None

        # The max time per task, unit: second
        # NOTE: The task will be stopped if the time is reached or done.
        self.max_time_per_task = self.config.experiment.auto_tuner.control.get(
            "max_time_per_task", 300
        )

        # The max time of auto tuner, if None, no limit.
        self.max_time = self.config.experiment.auto_tuner.control.get("max_time", None)

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

            if (
                self.cur_strategy["performance"]
                and self.config.experiment.auto_tuner.platform.get("airs_switch", False)
                and not self.has_checkout
            ):
                self.checkout()

            # get best strategy
            best_strategy = self.get_best()
            if best_strategy:
                self.logger.info(
                    f"Best strategy tuned so far: {best_strategy}, and performance is {best_strategy['performance']}."
                )
            else:
                self.logger.info(f"No strategy can run so far.")
        tuner_end_time = time.time()
        self.logger.info(f"AutoTuner Ended in {tuner_end_time - tuner_start_time} seconds.")

        # Run the best task
        if self.config.experiment.auto_tuner.control.get("run_best", True):
            best_strategy = self.get_best()
            if best_strategy:
                self.logger.info(f"Run best Strategy: {best_strategy}")
            else:
                raise ValueError(f"No strategy can run.")
            best_task = self.generator.gen_best_task(best_strategy, self.orig_config)
            best_task.action = "run"
            runner = SSHTrainRunner(best_task)
            runner.run(monitor=True, interval=60)

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
        while strategy and (self.pruner is not None and self.pruner.prune(strategy, self.history)):
            strategy = self.searcher.search()
        if strategy:
            self.idx += 1
            strategy["idx"] = self.idx
            pruned_count = self.pruner.pruned_count if self.pruner is not None else 0
            pruned_by_memory_model = (
                self.pruner.pruned_by_memory_model if self.pruner is not None else 0
            )
            if "memory_model" in self.config.experiment.auto_tuner:
                self.logger.info(
                    f"Searching {self.idx+pruned_count} / {len(self.searcher.strategies)} strategy, Pruned {pruned_count} strategy, {pruned_by_memory_model} by memory model."
                )
            else:
                self.logger.info(
                    f"Searching {self.idx+pruned_count} / {len(self.searcher.strategies)} strategy, Pruned {pruned_count} strategy."
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
        self.runner = SSHTrainRunner(task)
        self.runner.run()
        # set start time
        self.task_start_time = time.time()

    def monitor(self):
        """Monitor the task until task timeout or completed."""
        # Sleep 3s to ensure the task is started
        time.sleep(3)
        running = False
        sub_process_running = False
        while True:
            # If the task timeout, stop monitoring
            end_time = time.time()
            # To increase the time to 600s for the first task with data processing and cache.
            if self.idx == 1:
                max_time_per_task = 2 * self.max_time_per_task
            else:
                max_time_per_task = self.max_time_per_task
            if end_time - self.task_start_time > max_time_per_task:
                self.runner.stop()
                self.cur_strategy["stopped_by_tuner"] = True
                break
            # If the task is completed or idle, stop monitoring
            try:
                status = self.runner._query_status()
                self.logger.info(f"task_{self.cur_strategy['idx']} status: {status.name}")
                if status == JobStatus.COMPLETED_OR_IDLE:
                    break
                if status == JobStatus.RUNNING:
                    running = True
                if status == JobStatus.TRANSITIONAL:
                    if running:
                        self.runner.stop()
                        break

                # Add sub process monitor
                sub_process = self.runner._query_sub_process_status()
                if sub_process:
                    sub_process_running = True

                elif not sub_process:
                    if sub_process_running:
                        self.logger.info("Sub process not working, stop the task.")
                        self.runner.stop()
                        self.cur_strategy["stopped_by_tuner"] = True
                        break

            except Exception as e:
                self.logger.info(e)
                time.sleep(self.interval)
            time.sleep(self.interval)

        end_time = time.time()

        # Add elapsed time
        self.cur_strategy["elapsed_time"] = round(end_time - self.task_start_time, 2)
        # Add start time
        readable_task_start_time = datetime.datetime.fromtimestamp(self.task_start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.cur_strategy["start_time"] = readable_task_start_time

        self.logger.info(
            "task_{} monitor time: {:.2f}s".format(
                self.cur_strategy["idx"], self.cur_strategy["elapsed_time"]
            )
        )

    def record(self):
        """Record the task result to csv"""
        self.recorder.record(self.cur_task, self.cur_strategy)
        self.recorder.save(self.history)

    def get_best(self):
        sorted_history = self.recorder.sort(self.history)
        if sorted_history and sorted_history[0] and sorted_history[0]["performance"]:
            return sorted_history[0]
        return None


class ServeAutoTunner(AutoTuner):
    def __init__(self, config: DictConfig):
        # Set logger
        OmegaConf.set_struct(config, False)
        logger = logging.getLogger("FlagScale-AutoTuner")
        logger.setLevel(logging.INFO)

        dir_path = os.path.join(config.experiment.exp_dir, "auto_tuner")
        log_path = os.path.join(dir_path, "tuner.log")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        handler = logging.FileHandler(log_path, mode="a+")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger
        self.handler = handler
        deploy_config = config.experiment.get("runner", {}).get("deploy", {})

        if not deploy_config.get("use_fs_serve", True) and deploy_config.get("port", None):
            for item in config.serve:
                if item.get("serve_id") == "vllm_model":
                    item.engine_args["port"] = (
                        config.experiment.get("runner", {}).get("deploy", {}).get("port", None)
                    )

        # Deepcopy the original config to isolate from each task config
        # Modify the orig config when run best task
        self.orig_config = config
        self.config = copy.deepcopy(config)
        self.interval = self.config.experiment.auto_tuner.control.get("interval", 10)

        # Set config of auto tuner
        if "auto_tuner" not in self.config.experiment:
            self.config.experiment.auto_tuner = {}

        # Add nodes, nproc_per_node, cards to build search space or prune
        assert "experiment" in config, "experiment is not in yaml file."
        assert "runner" in config.experiment, "runner is not in yaml file."
        nnodes = config.experiment.runner.get("nnodes", 1)
        nproc_per_node = config.experiment.runner.get("nproc_per_node", 8)
        self.config.experiment.auto_tuner.nproc_per_node = nproc_per_node
        self.config.experiment.auto_tuner.nnodes = nnodes

        # If on other platform, set the platform envs

        self.config.experiment.auto_tuner.cards = (
            self.config.experiment.auto_tuner.nnodes
            * self.config.experiment.auto_tuner.nproc_per_node
        )

        # Build core sub modules, such as Searcher, Pruner, Generator and Recorder
        self.searcher = ServeSearcher(self.config)
        self.pruner = None
        self.generator = ServeGenerator(self.config)
        self.recorder = ServeRecorder(self.config)

        # Each task has its own runner
        self.runner = None

        # The max time per task, unit: second
        # NOTE: The task will be stopped if the time is reached or done.
        self.max_time_per_task = self.config.experiment.auto_tuner.control.get(
            "max_time_per_task", 3000
        )

        # The max time of auto tuner, if None, no limit.
        self.max_time = self.config.experiment.auto_tuner.control.get("max_time", None)

        # The start time of each task, used to control each task when stop
        self.start_task_time = None

        # The start time of tuner, used to control the tuner when stop
        self.start_time = time.time()

        # History strategy
        self.history = []
        self.cur_result = None

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
            self.logger.addHandler(self.handler)
            self.gen()
            if not self.cur_strategy:
                break
            self.logger.info(f"Run task_{self.idx}: {self.cur_strategy}")
            self.run()
            self.logger.info(f"Monitor task_{self.idx}:")
            self.monitor()
            self.logger.info(f"Record task_{self.idx}:")
            self.record()

            # get best strategy
            best_strategy = self.get_best()
            if best_strategy:
                self.logger.info(
                    f"Best strategy tuned so far: {best_strategy}, and {self.recorder.metric} is {best_strategy[self.recorder.metric]}."
                )
            else:
                self.logger.info(f"No strategy can run so far.")
        tuner_end_time = time.time()
        self.logger.info(f"AutoTuner Ended in {tuner_end_time - tuner_start_time} seconds.")

        # Run the best task
        if self.config.experiment.auto_tuner.control.get("run_best", True):
            best_strategy = self.get_best()
            if best_strategy:
                self.logger.info(f"Run best Strategy: {best_strategy}")
            else:
                raise ValueError(f"No strategy can run.")
            best_task = self.generator.gen_best_task(best_strategy, self.orig_config)
            best_task.action = "run"
            runner = SSHServeRunner(best_task)
            runner.run()

    def run(self, task=None):
        # Instantiate a runner and run the task
        if task is None:
            task = self.cur_task
        self.runner = SSHServeRunner(task)
        self.runner.run()
        # set start time
        self.task_start_time = time.time()

    def monitor(self):
        """Monitor the task until task timeout or completed."""
        # Sleep 3s to ensure the task is started
        time.sleep(3)
        running = False
        serve_alive = False
        while True:
            # If the task timeout, stop monitoring
            end_time = time.time()
            # To increase the time to 600s for the first task with data processing and cache.
            if self.idx == 1:
                max_time_per_task = 2 * self.max_time_per_task
            else:
                max_time_per_task = self.max_time_per_task
            if end_time - self.task_start_time > max_time_per_task:
                self.runner.stop()
                self.cur_strategy["stopped_by_tuner"] = True
                break
            # If the task is completed or idle, stop monitoring
            try:
                status = self.runner._query_status()
                if running:
                    serve_alive = self.runner._serve_alive()
                    if serve_alive:
                        break
                self.logger.info(f"task_{self.cur_strategy['idx']} status: {status.name}")
                if status == JobStatus.COMPLETED_OR_IDLE:
                    break
                if status == JobStatus.RUNNING:
                    running = True
                if status == JobStatus.TRANSITIONAL:
                    if running:
                        self.runner.stop()
                        break

            except Exception as e:
                self.logger.info(e)
                time.sleep(self.interval)
            time.sleep(self.interval)

        if serve_alive:
            try:
                result = self.runner._profile_serve()
                self.cur_result = result
            except Exception as e:
                self.logger.info(f"fail to get profile result {e}")
        time.sleep(self.interval)
        if running:
            self.runner.stop()

        end_time = time.time()

        # Add elapsed time
        self.cur_strategy["elapsed_time"] = round(end_time - self.task_start_time, 2)
        # Add start time
        readable_task_start_time = datetime.datetime.fromtimestamp(self.task_start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.cur_strategy["start_time"] = readable_task_start_time

        self.logger.info(
            "task_{} monitor time: {:.2f}s".format(
                self.cur_strategy["idx"], self.cur_strategy["elapsed_time"]
            )
        )

    def record(self):
        self.recorder.record(self.cur_strategy, self.cur_result)
        self.history.append(self.recorder.cur_strategy)
        self.recorder.save(self.history)

    def get_best(self):
        sorted_history = self.recorder.sort(self.history)
        if sorted_history and sorted_history[0] and sorted_history[0][self.recorder.metric]:
            return sorted_history[0]
        return None
