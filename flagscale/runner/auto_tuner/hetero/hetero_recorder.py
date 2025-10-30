import csv
import logging

import pandas as pd

from flagscale.runner.auto_tuner.record.recorder import Recorder


class HeteroRecorder(Recorder):
    """
    Recorder for heterogeneous tasks.

    This class reuses the core log-parsing and sorting logic from the base
    Recorder. It overrides the `save` method to provide robust handling
    of the history list to prevent data loss or duplication during saving,
    and to ensure correct CSV formatting for complex data types.
    """

    def __init__(self, config):
        """Initializes the HeteroRecorder."""
        super().__init__(config)

    def record(self, task, strategy):
        """
        Records the task results and sanitizes error messages.
        """
        # Call the parent method to grep logs for performance, mem, errors
        super().record(task, strategy)

        # Sanitize error strings to prevent CSV line breaks
        if strategy.get("error"):
            error_string = str(strategy["error"])
            # Replace newlines with spaces to keep CSV row intact
            sanitized_error = error_string.replace("\n", " ").replace("\r", "")
            strategy["error"] = sanitized_error

    def save(self, history: list):
        """
        Overrides the save method to be robust against empty history,
        duplicate indices, failed/pruned tasks, and complex data formatting.
        """
        if not history:
            self.logger.warning("History is empty, creating an empty history.csv file.")
            pd.DataFrame().to_csv(self.path, index=False)
            return

        # Create DataFrame from the *raw* history list *before* sorting.
        # This ensures failed or pruned tasks (which might be filtered by self.sort) are included.
        try:
            df = pd.DataFrame(history)
        except Exception as e:
            self.logger.error(f"Error creating DataFrame from raw history: {e}. Saving empty file.")
            pd.DataFrame().to_csv(self.path, index=False)
            return

        if df.empty:
            self.logger.warning("DataFrame is empty after creation. Saving empty file.")
            df.to_csv(self.path, index=False)
            return

        # Sort the DataFrame using pandas, which handles None/NaN values correctly.
        try:
            if 'performance' in df.columns:
                # Ensure performance column is numeric for a safe sort
                df['performance'] = pd.to_numeric(df['performance'], errors='coerce')
                ascending_order = self.sorted_order == "ascend"
                # na_position='last' ensures failed tasks (performance=NaN) are sorted to the end
                df.sort_values(
                    by='performance', ascending=ascending_order, na_position='last', inplace=True
                )
        except Exception as e:
            self.logger.warning(f"Could not sort history DataFrame: {e}. Saving unsorted.")

        # Deduplication & Column Management
        cols = df.columns.tolist()
        if 'idx' in cols:
            try:
                # Drop duplicate idx entries, keep the last (most recent) one
                initial_rows = len(df)
                df.drop_duplicates(subset=['idx'], keep='last', inplace=True)
                rows_after_dedup = len(df)
                if initial_rows > rows_after_dedup:
                    self.logger.warning(
                        f"Removed {initial_rows - rows_after_dedup} duplicate idx entries before saving."
                    )

                # Re-fetch columns after drop
                cols = df.columns.tolist()
                # Move 'idx' to the first column
                cols.insert(0, cols.pop(cols.index("idx")))
                df = df.reindex(columns=cols)
            except ValueError:
                self.logger.warning("Could not reorder 'idx' column.")
        else:
            self.logger.warning(
                f"Could not find 'idx' column for sorting/deduplication, saving with default order. Keys: {cols}"
            )

        # Drop internal-use columns that shouldn't be in the final CSV
        columns_to_drop = []
        # 'stopped_by_tuner' is an internal flag
        if "stopped_by_tuner" in cols:
            columns_to_drop.append("stopped_by_tuner")
        # 'pruned' is an internal flag that caused format errors
        if "pruned" in cols:
            columns_to_drop.append("pruned")
        # 'prune_reason' is internal debug info
        if "prune_reason" in cols:
            columns_to_drop.append("prune_reason")

        if columns_to_drop:
            try:
                df = df.drop(columns=columns_to_drop)
            except KeyError as e:
                self.logger.warning(f"Could not drop columns: {e}. Skipping drop.")

        try:
            # Robust formatting for complex fields (like lists/meshes) ---
            # Convert object columns (like lists) to well-formatted JSON strings
            for column in df.columns:
                # Check if column dtype is 'object' (often strings, lists, dicts)
                if df[column].dtype == 'object':
                    # self.to_str is inherited from base Recorder
                    df[column] = df[column].map(self.to_str)

            # Use quoting=csv.QUOTE_ALL to wrap all fields in quotes,
            # preventing commas in lists/strings from breaking columns.
            df.to_csv(self.path, index=False, quoting=csv.QUOTE_ALL)
            self.logger.info(f"Saved {len(df)} unique strategy records to {self.path}")
        except Exception as e:
            self.logger.error(f"Error saving history to CSV: {e}")
