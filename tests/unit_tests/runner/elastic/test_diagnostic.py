import os
import tempfile

from unittest.mock import MagicMock, mock_open, patch

import pytest

from flagscale.runner.elastic.diagnostic import error_types, generate_diagnostic_report


class TestDiagnostic:
    """Test cases for diagnostic module"""

    @pytest.fixture
    def mock_config(self):
        """Mock config object"""
        return MagicMock()

    @pytest.fixture
    def sample_log_content(self):
        """Sample log content with various errors"""
        return """
        [INFO] Starting training...
        [ERROR] CUDA out of memory
        Traceback (most recent call last):
          File "train.py", line 100, in <module>
            model.forward()
        torch.distributed.elastic.rendezvous.api.RendezvousConnectionError: Connection failed
        OutOfMemoryError: GPU memory exhausted
        [ERROR] Training failed
        """

    def test_error_types_dict_exists(self):
        """Test that error_types dictionary is properly defined"""
        assert isinstance(error_types, dict)
        assert len(error_types) > 0

        expected_keys = ['out of memory', 'rendezvous', 'traceback', 'error', 'cuda']
        for key in expected_keys:
            assert key in error_types

    def test_generate_diagnostic_report_empty_file(self, mock_config):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )
            assert report == ""
            assert "Diagnostic Report for localhost (node 0)" in report
            assert "Log file is empty" in report
        finally:
            os.unlink(temp_path)

    def test_generate_diagnostic_report_with_errors(self, mock_config, sample_log_content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(sample_log_content)
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )

            assert "Diagnostic Report for localhost (node 0)" in report
            assert "OutOfMemoryError" in report
            assert "RendezvousConnectionError" in report
            assert "CodeError" in report
            assert "GeneralError" in report
        finally:
            os.unlink(temp_path)

    def test_generate_diagnostic_report_nonexistent_file(self, mock_config):
        report = generate_diagnostic_report(
            mock_config, "localhost", 0, "/nonexistent/file.log", return_content=True
        )

        assert "Diagnostic Report for localhost (node 0)" in report
        assert "Log file is empty or does not exist" in report

    def test_generate_diagnostic_report_no_errors(self, mock_config):
        content = """
        [INFO] Training started successfully
        [INFO] Epoch 1/100 completed
        [INFO] Training finished successfully
        """

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )

            assert "Diagnostic Report for localhost (node 0)" in report
            assert "No errors or unknown error detected" in report
        finally:
            os.unlink(temp_path)

    def test_generate_diagnostic_report_file_output(self, mock_config, sample_log_content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(sample_log_content)
            temp_path = f.name

        try:
            # Test file output mode
            result_path = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=False
            )

            assert result_path is not None
            assert "diagnostic" in result_path or result_path.endswith('.txt')
        finally:
            os.unlink(temp_path)

    @patch('flagscale.runner.elastic.diagnostic.logger')
    def test_generate_diagnostic_report_read_error(self, mock_logger, mock_config):
        """Test diagnostic report generation with file read error"""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, "/some/file.log", return_content=True
            )

            assert "Error reading log file" in report
            mock_logger.error.assert_called()

    def test_error_detection_case_insensitive(self, mock_config):
        content = """
        CUDA OUT OF MEMORY ERROR occurred
        TRACEBACK: Failed to execute
        RENDEZVOUS connection failed
        """

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )

            # Should detect errors regardless of case
            assert "OutOfMemoryError" in report
            assert "CodeError" in report
            assert "RendezvousError" in report
        finally:
            os.unlink(temp_path)

    def test_multiple_error_types_detection(self, mock_config):
        content = """
        CUDA error: out of memory
        ImportError: module not found
        Permission denied: cannot access file
        Connection timeout occurred
        Process was killed
        """

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )

            assert "CUDAError" in report
            assert "ImportError" in report
            assert "PermissionError" in report
            assert "TimeoutError" in report
            assert "ProcessKilled" in report
        finally:
            os.unlink(temp_path)
