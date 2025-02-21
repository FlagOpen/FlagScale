import math

import torch


class SpikyLossDetector:
    """This class represents a Spiky Loss Detector.
    It is used to detect spikes in loss values during training.
    """

    def __init__(self, threshold=0.2, loss=None):
        self.last_loss = loss
        self.threshold = threshold

    def reduce_losses(self, losses_reduced):
        loss_reduced = {}
        from megatron.core import mpu

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator
        return loss_reduced.get("lm loss")

    def is_spkiy_loss(self, loss):
        if loss is None:
            return False
        if self.last_loss is not None:
            if math.isnan(loss) or math.isnan(self.last_loss):
                self.last_loss = loss
            elif math.isinf(loss) or math.isinf(self.last_loss):
                return True
            else:
                result = (loss - self.last_loss) / self.last_loss >= self.threshold
                if result:
                    return True
                else:
                    self.last_loss = loss
        else:
            self.last_loss = loss
        return False
