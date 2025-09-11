import os
import signal
import sys
import threading
import time

from datetime import datetime
from typing import Any, Dict, Optional

from flagscale.runner.elastic.diagnostic import generate_diagnostic_report
from flagscale.runner.elastic.log_collector import collect_logs
from flagscale.runner.runner_base import JobStatus
from flagscale.runner.utils import logger


class MonitorService:
    """
    An independent monitoring service for background monitoring of training task status, log collection, and diagnostic report generation.
    """

    def __init__(self, config, runner_instance, interval=10):
        """
        Initializing service

        Args:
            config: Configuration object
            runner_instance: runner instance
            interval: interval time
        """
        self.config = config
        self.runner = runner_instance
        self.interval = interval
        self.is_running = False
        self.monitor_thread = None
        self.log_collection_enabled = True
        self.diagnostic_enabled = True

        self.monitor_log_dir = os.path.join(config.train.system.logging.log_dir, "monitor")
        os.makedirs(self.monitor_log_dir, exist_ok=True)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, stopping monitor service...")
        self.stop()
        sys.exit(0)

    def start_monitoring(self, enable_log_collection=True, enable_diagnostic=True):
        """
        Start monitoring service (non-blocking)

        Args:
            enable_log_collection: Whether to enable log collection
            enable_diagnostic: Whether to enable diagnostic report generation
        """
        if self.is_running:
            logger.warning("Monitor service is already running")
            return

        self.log_collection_enabled = enable_log_collection
        self.diagnostic_enabled = enable_diagnostic
        self.is_running = True

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Monitor service started with interval={self.interval}s")
        logger.info(f"Log collection enabled: {enable_log_collection}")
        logger.info(f"Diagnostic enabled: {enable_diagnostic}")
        logger.info(f"Monitor logs will be saved to: {self.monitor_log_dir}")

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        logger.info("Monitor service stopped")

    def _monitor_loop(self):
        logger.info("Starting monitoring loop...")

        time.sleep(self.interval)

        try:
            while self.is_running:
                start_time = time.time()

                try:
                    job_status = self._get_job_status()
                    logger.info(f"Job Status: {job_status.name}")

                    self._log_status(job_status)

                    if job_status == JobStatus.COMPLETED_OR_IDLE:
                        logger.info("Job completed, stopping monitoring")
                        break

                    if self.log_collection_enabled:
                        self._collect_logs()

                    if self.diagnostic_enabled:
                        self._generate_diagnostics()

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)

                if self.is_running:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Monitor loop crashed: {e}")
        finally:
            logger.info("Monitor loop ended")
            self.is_running = False

    def _get_job_status(self) -> JobStatus:
        return self.runner._query_status()

    def _log_status(self, status: JobStatus):
        status_log_file = os.path.join(self.monitor_log_dir, "status.log")

        try:
            with open(status_log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Status: {status.name}\n")
        except Exception as e:
            logger.error(f"Failed to write status log: {e}")

    def _collect_logs(self):
        if not hasattr(self.runner, 'resources') or self.runner.resources is None:
            self._collect_logs_for_host("localhost", 0)
        else:
            for node_rank, (host, _) in enumerate(self.runner.resources.items()):
                self._collect_logs_for_host(host, node_rank)

    def _collect_logs_for_host(self, host: str, node_rank: int):
        try:
            log_file = collect_logs(
                self.config, host, node_rank, self.monitor_log_dir, dryrun=False
            )

            if log_file:
                logger.debug(f"Collected logs for {host} (node {node_rank}): {log_file}")
        except Exception as e:
            logger.error(f"Failed to collect logs for {host} (node {node_rank}): {e}")

    def _generate_diagnostics(self):
        if not hasattr(self.runner, 'resources') or self.runner.resources is None:
            self._generate_diagnostic_for_host("localhost", 0)
        else:
            for node_rank, (host, _) in enumerate(self.runner.resources.items()):
                self._generate_diagnostic_for_host(host, node_rank)

    def _generate_diagnostic_for_host(self, host: str, node_rank: int):
        try:
            log_files = [
                f
                for f in os.listdir(self.monitor_log_dir)
                if f.startswith(f"host_{node_rank}_{host}_temp_") and f.endswith(".log")
            ]

            if log_files:
                latest_log = max(
                    log_files, key=lambda f: os.path.getmtime(os.path.join(self.monitor_log_dir, f))
                )
                log_file_path = os.path.join(self.monitor_log_dir, latest_log)

                diagnostic_file = generate_diagnostic_report(
                    self.config, host, node_rank, log_file_path, return_content=False
                )

                if diagnostic_file:
                    logger.debug(
                        f"Generated diagnostic for {host} (node {node_rank}): {diagnostic_file}"
                    )
        except Exception as e:
            logger.error(f"Failed to generate diagnostic for {host} (node {node_rank}): {e}")

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "interval": self.interval,
            "log_collection_enabled": self.log_collection_enabled,
            "diagnostic_enabled": self.diagnostic_enabled,
            "monitor_log_dir": self.monitor_log_dir,
            "thread_alive": self.monitor_thread.is_alive() if self.monitor_thread else False,
        }


def main():
    """
    python monitor_service.py [config_file] [interval]
    """
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Run FlagScale monitor service")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--interval", type=int, default=10, help="Monitor interval in seconds")
    parser.add_argument("--no-log-collection", action="store_true", help="Disable log collection")
    parser.add_argument("--no-diagnostic", action="store_true", help="Disable diagnostic reports")

    args = parser.parse_args()

    if not args.config:
        logger.error("Config file is required")
        sys.exit(1)

    try:
        config = OmegaConf.load(args.config)

        # Here needs to create a runner instance according to the actual situation
        logger.info("Monitor service is designed to be integrated with runner_train.py")
        logger.info("For standalone usage, additional runner initialization is needed")

    except Exception as e:
        logger.error(f"Failed to start monitor service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
