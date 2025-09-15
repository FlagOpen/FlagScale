import os
import tempfile

from unittest.mock import MagicMock, call, patch

import pytest

from omegaconf import OmegaConf

from flagscale.runner.elastic.log_collector import _log_offsets, collect_logs


class TestLogCollector:
    """Test cases for log collector module"""

    @pytest.fixture
    def mock_config(self):
        return OmegaConf.create(
            {
                'train': {'system': {'logging': {'log_dir': '/tmp/test_logs'}}},
                'experiment': {'runner': {'no_shared_fs': False, 'ssh_port': 22}},
            }
        )

    @pytest.fixture
    def mock_config_no_shared_fs(self):
        return OmegaConf.create(
            {
                'train': {'system': {'logging': {'log_dir': '/tmp/test_logs'}}},
                'experiment': {'runner': {'no_shared_fs': True, 'ssh_port': 22}},
            }
        )

    def setup_method(self):
        """Reset log offsets before each test"""
        _log_offsets.clear()

    def test_collect_logs_localhost_initial(self, mock_config):
        """Test initial log collection from localhost"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Initial log content\nLine 2\nLine 3\n")
            src_log_path = f.name

        # Mock the source log file path
        expected_src = "/tmp/test_logs/host_0_localhost.output"

        with patch('os.path.join', return_value=expected_src):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=100):
                    with patch('os.makedirs'):
                        with patch(
                            'flagscale.runner.elastic.log_collector.run_local'
                        ) as mock_run_local:
                            result = collect_logs(
                                mock_config, "localhost", 0, "/tmp/dest", dryrun=False
                            )

                            # Should call run_local for localhost
                            mock_run_local.assert_called()

                            # Should return a destination file path
                            assert result is not None
                            assert "host_0_localhost_temp_" in result
                            assert result.endswith(".log")

        # Cleanup
        try:
            os.unlink(src_log_path)
        except:
            pass

    def test_collect_logs_localhost_incremental(self, mock_config):
        """Test incremental log collection"""
        # Set initial offset
        _log_offsets["localhost_0"] = 50

        expected_src = "/tmp/test_logs/host_0_localhost.output"

        with patch('os.path.join', return_value=expected_src):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=200):
                    with patch('os.makedirs'):
                        with patch(
                            'flagscale.runner.elastic.log_collector.run_local'
                        ) as mock_run_local:
                            result = collect_logs(
                                mock_config, "localhost", 0, "/tmp/dest", dryrun=False
                            )

                            mock_run_local.assert_called()
                            args, kwargs = mock_run_local.call_args
                            command = args[0]
                            assert "tail -c +51" in command  # offset + 1

                            assert _log_offsets["localhost_0"] == 200

    def test_collect_logs_remote_host(self, mock_config):
        expected_src = "/tmp/test_logs/host_0_worker1.output"

        with patch('os.path.join', return_value=expected_src):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=100):
                    with patch('os.makedirs'):
                        with patch(
                            'flagscale.runner.elastic.log_collector.run_scp'
                        ) as mock_run_scp:
                            result = collect_logs(
                                mock_config, "worker1", 0, "/tmp/dest", dryrun=False
                            )

                            mock_run_scp.assert_called_with(
                                "worker1", expected_src, result, 22, False, incremental=True
                            )

    def test_collect_logs_no_shared_fs(self, mock_config_no_shared_fs):
        expected_src = "/tmp/test_logs/host.output"

        with patch('os.path.join', return_value=expected_src):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=100):
                    with patch('os.makedirs'):
                        with patch(
                            'flagscale.runner.elastic.log_collector.run_local'
                        ) as mock_run_local:
                            result = collect_logs(
                                mock_config_no_shared_fs, "localhost", 0, "/tmp/dest", dryrun=False
                            )

                            mock_run_local.assert_called()

    def test_collect_logs_file_not_found(self, mock_config):
        with patch('os.path.join'):
            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs'):
                    with patch('flagscale.runner.elastic.log_collector.logger') as mock_logger:
                        result = collect_logs(
                            mock_config, "localhost", 0, "/tmp/dest", dryrun=False
                        )

                        assert result is None
                        mock_logger.warning.assert_called()

    def test_collect_logs_empty_file(self, mock_config):
        with patch('os.path.join'):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=0):
                    with patch('os.makedirs'):
                        with patch('flagscale.runner.elastic.log_collector.logger') as mock_logger:
                            result = collect_logs(
                                mock_config, "localhost", 0, "/tmp/dest", dryrun=False
                            )

                            assert result is None
                            mock_logger.warning.assert_called()

    def test_collect_logs_dryrun(self, mock_config):
        with patch('os.path.join'):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=100):
                    with patch('os.makedirs'):
                        with patch(
                            'flagscale.runner.elastic.log_collector.run_local'
                        ) as mock_run_local:
                            result = collect_logs(
                                mock_config, "localhost", 0, "/tmp/dest", dryrun=True
                            )

                            mock_run_local.assert_called()
                            args, kwargs = mock_run_local.call_args
                            assert kwargs['dryrun'] == True

    def test_collect_logs_exception_handling(self, mock_config):
        with patch('os.path.join'):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=100):
                    with patch('os.makedirs'):
                        with patch(
                            'flagscale.runner.elastic.log_collector.run_local',
                            side_effect=Exception("Test error"),
                        ):
                            with patch(
                                'flagscale.runner.elastic.log_collector.logger'
                            ) as mock_logger:
                                result = collect_logs(
                                    mock_config, "localhost", 0, "/tmp/dest", dryrun=False
                                )

                                assert result is None
                                mock_logger.error.assert_called()

    def test_log_offsets_management(self, mock_config):
        assert "localhost_0" not in _log_offsets

        with patch('os.path.join'):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=100):
                    with patch('os.makedirs'):
                        with patch('flagscale.runner.elastic.log_collector.run_local'):
                            collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)
                            assert _log_offsets["localhost_0"] == 100

                            with patch('os.path.getsize', return_value=200):
                                collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)
                                assert _log_offsets["localhost_0"] == 200

    def test_destination_file_cleanup_on_failure(self, mock_config):
        dest_file = "/tmp/dest/host_0_localhost_temp_test.log"

        with patch('os.path.join'):
            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs'):
                    with patch('os.path.exists') as mock_exists:
                        with patch('os.remove') as mock_remove:
                            # Mock that dest file exists
                            mock_exists.side_effect = lambda path: path == dest_file

                            result = collect_logs(
                                mock_config, "localhost", 0, "/tmp/dest", dryrun=False
                            )

                            assert result is None
