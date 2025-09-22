import logging

import pandas as pd

from flagscale.runner.auto_tuner.record.recorder import Recorder


class HeteroRecorder(Recorder):
    """
    Recorder for heterogeneous tasks. This class exists to maintain architectural
    consistency. It inherits directly from the base Recorder as logging formats
    are currently identical for both homogeneous and heterogeneous runs.
    """

    def __init__(self, config):
        super().__init__(config)

    def save(self, history: list):
        """
        Overrides the save method to be robust against
        edge cases like empty history or missing columns.
        """
        if not history:
            self.logger.warning("History is empty, creating an empty history.csv file.")
            pd.DataFrame().to_csv(self.path, index=False)
            return

        sorted_history = self.sort(history)

        if not sorted_history:
            self.logger.warning("All tasks were pruned; saving pruned strategies to history.")
            df = pd.DataFrame(history)
        else:
            df = pd.DataFrame(sorted_history)

        if 'idx' in df.columns:
            cols = df.columns.tolist()
            if cols:
                cols.insert(0, cols.pop(cols.index("idx")))
                df = df.reindex(columns=cols)
        else:
            self.logger.warning(
                f"Could not find 'idx' column for sorting, saving with default order. Keys: {df.columns.tolist()}"
            )

        if "stopped_by_tuner" in df.columns:
            df = df.drop(columns=["stopped_by_tuner"])

        df.to_csv(self.path, index=False, escapechar="\\")
        self.logger.info(f"Saved {len(df)} strategy records to {self.path}")
