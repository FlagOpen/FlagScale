import pandas as pd

from ..record.recorder import Recorder


class HeteroRecorder(Recorder):
    def __init__(self, config):
        super().__init__(config)

    def save(self, history):

        sorted_history = self.sort(history)

        processed_history = []
        if not sorted_history:
            return

        for s in sorted_history:
            s_copy = s.copy()
            for key, value in s_copy.items():
                if isinstance(value, list):
                    s_copy[key] = str(value)
            processed_history.append(s_copy)

        df = pd.DataFrame(processed_history)

        cols = df.columns.tolist()
        if "idx" in cols:
            cols.insert(0, cols.pop(cols.index("idx")))
            df = df.reindex(columns=cols)
        if "stopped_by_tuner" in df.columns:
            df = df.drop(columns=["stopped_by_tuner"])

        df.to_csv(self.path, index=False, escapechar="\\")
