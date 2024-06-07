import os
import re
import logging
import subprocess
import pandas as pd


class Recorder:

    def __init__(self, config):
        self.config = config
        self.path = os.path.join(
            config.experiment.exp_dir,
            "auto_tuner",
            "history.csv",
        )
        # Metric to grep in the last rank of last node log file
        if (
            "auto_tuner" in self.config
            and "performance" in self.config.experiment.auto_tuner
        ):
            self.metric = self.config.experiment.auto_tuner.performance.get(
                "name", "elapsed time per iteration \(ms\):"
            )
        else:
            self.metric = "elapsed time per iteration \(ms\):"

        # Sort order of performance, order just in [ascend, and descend], default ascend
        if (
            "auto_tuner" in self.config
            and "performance" in self.config.experiment.auto_tuner
        ):
            self.sorted_order = self.config.experiment.auto_tuner.performance.get(
                "order", "ascend"
            )
        else:
            self.sorted_order = "ascend"

        self.logger = logging.getLogger("FlagScale-AutoTuner")
        self.cur_strategy = None

    def record(self, task, strategy):
        """Record the performance and max memory of task"""
        self.cur_strategy = strategy
        peformance_path, host_path = self.get_performance_and_host_path(task)

        errors = self.grep_error(host_path)
        if errors:
            # If OOM in errors, the task must fail.
            if "OOM" in errors:
                strategy["performance"] = None
                strategy["max_mem"] = "OOM"
                strategy["error"] = "|".join(list(errors))

            # If task is stopped by autotuner, task may not be failed,just hang or too slow.
            elif self.cur_strategy.get("stopped_by_tuner", False):
                performace = self.grep_performance(peformance_path, self.metric)
                strategy["performance"] = performace
                strategy["max_mem"] = self.grep_max_memory(host_path)
                strategy["error"] = None

            # Task failed and the code may have logical errors
            else:
                strategy["performance"] = None
                strategy["max_mem"] = self.grep_max_memory(host_path)
                strategy["error"] = "|".join(list(errors))

        # Task ended properly
        else:
            strategy["max_mem"] = self.grep_max_memory(host_path)
            performace = self.grep_performance(peformance_path, self.metric)
            strategy["performance"] = performace
            strategy["error"] = None

        # Pass back to platform if need
        if (
            "airs_switch" in self.config.experiment.auto_tuner.platform
            and self.config.experiment.auto_tuner.platform.airs_switch
            and strategy["performance"]
        ):
            self.pass_back_to_platform(strategy)

    def pass_back_to_platform(self, strategy):
        gbs = int(self.config.train.model.global_batch_size)
        seq_len = int(self.config.train.model.seq_length)
        throughput = gbs * seq_len / (strategy["performance"] / 1000)
        day = round(
            self.config.train.model.train_samples
            * seq_len
            / (throughput * 60 * 60 * 24),
            2,
        )
        command = [
            "airsctl job performance",
            "-D",
            f"{strategy['data_parallel_size']}",
            "--distributed_optimizer",
            (
                f"{strategy['use_distributed_optimizer']}"
                if strategy["use_distributed_optimizer"] is not None
                else "False"
            ),
            "-E",
            f"{strategy['expert_model_parallel_size']}",
            "-C",
            f"{strategy['context_parallel_size']}",
            "-M",
            f"{strategy['micro_batch_size']}",
            "-L",
            f"{strategy['pipeline_model_parallel_size']}",
            "-G",
            (
                f"{strategy['recompute_granularity']}"
                if strategy["recompute_granularity"]
                else "None"
            ),
            "-R",
            (
                f"{strategy['recompute_method']}"
                if strategy["recompute_granularity"]
                else "None"
            ),
            "-N",
            (
                f"{strategy['recompute_num_layers']}"
                if strategy["recompute_num_layers"]
                else "0"
            ),
            "-S",
            (
                f"{strategy['sequence_parallel']}"
                if strategy["sequence_parallel"] is not None
                else "False"
            ),
            "-T",
            f"{strategy['tensor_model_parallel_size']}",
            "-V",
            (
                f"{strategy['num_layers_per_virtual_pipeline_stage']}"
                if strategy["num_layers_per_virtual_pipeline_stage"]
                else "0"
            ),
            "--throughput",
            f"{int(throughput)}",
            "--day",
            f"{day}",
            "--time",
            f"{int(strategy['performance'])}",
        ]
        joined_command = " ".join(command)
        self.logger.info(f"Pass back to platform: {joined_command}")
        try:
            subprocess.run(joined_command, shell=True, check=True)
        except Exception as e:
            self.logger.info(f"Failed to pass back to platform: {e}")

    def grep_max_memory(self, path, pattern="max reserved"):
        """Read the log file and return the max memory."""
        if not os.path.exists(path):
            raise ValueError(f"The path do not exist: {path}")
        memory_pattern = pattern + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + pattern
        max_memory = None
        for item in os.listdir(path):
            if not item.startswith("host_") and not item.endswith(".output"):
                continue
            file_path = os.path.join(path, item)
            with open(file_path, "rb") as f:
                for _ in f:
                    try:
                        line = _.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    memory = re.findall(memory_pattern, line, re.IGNORECASE)
                    if memory:
                        value = None
                        for item in memory[0]:
                            try:
                                value = float(item)
                                if max_memory is None:
                                    max_memory = value
                                if value > max_memory:
                                    max_memory = value
                                break
                            except:
                                continue
                        assert value is not None, "Can't grep the max memory"
        self.logger.info(f"task_{self.cur_strategy['idx']} max_memory: {max_memory}")
        return max_memory

    def get_performance_and_host_path(self, task):
        """Get the log path of task."""
        logs = os.path.join(task.experiment.exp_dir, "logs")
        details = os.path.join(logs, "details")

        if not os.path.exists(details):
            raise ValueError("The task detail folder does not exist.")

        # The loss just logged in the last rank of the last node
        max_host_rank = 0
        max_host = None
        for item in os.listdir(details):
            if not item.startswith("host_"):
                continue
            host_rank = int(item.split("_")[1])
            if host_rank >= max_host_rank:
                max_host_rank = host_rank
                max_host = item
        if max_host is None:
            return None, logs
        outputs = os.listdir(os.path.join(details, max_host))
        assert len(outputs) == 1, f"the sub dir of {outputs} must be just one."
        new_outputs = os.listdir(os.path.join(details, max_host, outputs[0]))
        assert len(new_outputs) == 1, f"the sub dir of {new_outputs} must be just one."
        last_path = os.path.join(
            details, max_host, outputs[0], new_outputs[0], "attempt_0"
        )
        last_dir = None
        last_dir_rank = 0
        for item in os.listdir(last_path):
            try:
                rank = int(item)
                if rank > last_dir_rank:
                    last_dir = item
                    last_dir_rank = rank
            except Exception as e:
                raise e
        log_path = os.path.join(last_path, last_dir, "stdout.log")
        if not os.path.exists(log_path):
            raise ValueError("The log file does not exist.")
        return log_path, logs

    def grep_performance(self, path, pattern="elapsed time per iteration \(ms\):"):
        """Read the log file and return the performance."""
        metric_pattern = pattern + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + pattern
        if not path or not os.path.exists(path):
            return None
        performance = []
        with open(path, "rb") as f:
            for _ in f:
                try:
                    line = _.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                metric = re.findall(metric_pattern, line)
                if metric:
                    value = None
                    for item in metric[0]:
                        try:
                            value = float(item)
                            performance.append(value)
                            break
                        except:
                            continue
                    assert value is not None, "Can't grep the performance"
        if not performance:
            self.logger.info(f"task_{self.cur_strategy['idx']} performance: {None}")
            return None
        if len(performance) == 1:
            self.logger.info(
                f"task_{self.cur_strategy['idx']} performance: {performance[0]}"
            )
            return round(performance[0], 3)
        else:
            average = sum(performance[1:]) / (len(performance) - 1)
            self.logger.info(f"task_{self.cur_strategy['idx']} performance: {average}")
            return round(average, 3)

    def grep_error(self, path, pattern="Error"):
        """Read the log file and return the error"""
        if not os.path.exists(path):
            raise ValueError(f"The path do not exist: {path}")

        oom = "out of memory"
        errors_info = set()
        for item in os.listdir(path):
            if not item.startswith("host_") and not item.endswith(".output"):
                continue
            file_path = os.path.join(path, item)
            with open(file_path, "rb") as f:
                for _ in f:
                    try:
                        line = _.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    error = re.findall(pattern, line, re.IGNORECASE)
                    if error:
                        if oom in line:
                            errors_info.add("OOM")
                            errors_info.add(line)
                        else:
                            errors_info.add(line)

        self.logger.info(f"task_{self.cur_strategy['idx']} error: {errors_info}")
        return errors_info

    def sort(self, history):
        no_pruned_history = []
        for strategy in history:
            if not strategy.get("pruned", False):
                no_pruned_history.append(strategy)

        sorted_history = None
        if self.sorted_order == "ascend":
            sorted_history = sorted(
                no_pruned_history,
                key=lambda x: (
                    x["performance"] if x["performance"] is not None else float("inf")
                ),
            )
        elif self.sorted_order == "descend":
            sorted_history = sorted(
                no_pruned_history,
                key=lambda x: (
                    x["performance"] if x["performance"] is not None else float("-inf")
                ),
                reverse=True,
            )
        else:
            raise ValueError(f"The sorted order {self.sorted_order} is not supported.")
        assert sorted_history is not None
        return sorted_history

    def save(self, history):
        """Store history to csv file."""
        # sort history
        sorted_history = self.sort(history)
        df = pd.DataFrame(sorted_history)
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index("idx")))
        df = df.reindex(columns=cols)
        if "stopped_by_tuner" in df.columns:
            df = df.drop(columns=["stopped_by_tuner"])
        df.to_csv(self.path, index=False)
