from torch.distributed import get_rank
try:
    import nvidia_resiliency_ext.attribution.straggler as straggler
except ImportError:
    straggler = None

class StragglerDetectionWrapper:
    def __init__(self, level, section_name):
        self.level = level
        self.section_name = section_name

    def __call__(self, dest_func):
        def wrapper(*args, **kwargs):
            passed_warmup_stage = kwargs.pop('passed_warmup_stage', None)
            user_specified_level = kwargs.pop('user_specified_level', None)
            generate_report = kwargs.pop('generate_report', False)
            if passed_warmup_stage and user_specified_level and user_specified_level == self.level:
                if straggler is None:
                    raise RuntimeError(
                        "The 'nvidia_resiliency_ext' module is required for straggler "
                        "detection but was not found. Please ensure it is installed."
                    )
                res = None
                with straggler.Detector.detection_section(self.section_name, profile_cuda=True):
                    res = dest_func(*args, **kwargs)
                if generate_report:
                    report = straggler.Detector.generate_report()
                    rank = get_rank()
                    if rank == 0:
                        print(
                            f"Rank {rank} GPUs relative perf: {report.gpu_relative_perf_scores}"
                        )
                        print(
                            f"Rank {rank} GPUs individual perf: {report.gpu_individual_perf_scores}"
                        )
                        print(
                            f"Rank {rank} sections relative perf: {report.section_relative_perf_scores}"
                        )
                        print(
                            f"Rank {rank} sections individual perf: {report.section_individual_perf_scores}"
                        )

                        stragglers  = report.identify_stragglers(gpu_rel_threshold=0.99, gpu_indiv_threshold=0.99)
                        for key, straggler_ids in stragglers.items():
                            if key == 'straggler_gpus_individual' or key == 'straggler_gpus_relative':
                                straggler_ranks = [str(straggler_id.rank) for straggler_id in straggler_ids]
                                if straggler_ranks:
                                    straggler_ranks.sort()
                                    straggler_ranks = ','.join(straggler_ranks)
                                    print(
                                        f'Rank {straggler_ranks} identified as \'{key}\' straggler(s)'
                                    )
                                else:
                                    print(
                                        f'No rank is identified as as \'{key}\' straggler'
                                    )
                            elif key == 'straggler_sections_individual' or key == 'straggler_sections_relative':
                                # TODO: Currently, NVRx simply monitors CPU times of sections, resulting in unpredictable output.
                                pass
                return res
            return dest_func(*args, **kwargs)
        return wrapper

def initialize_straggler_detector(user_specified_level):
    if user_specified_level > 0:
        if not straggler:
            raise RuntimeError(
                "The 'nvidia_resiliency_ext' module is required for straggler "
                "detection but was not found. Please ensure it is installed."
            )
        straggler.Detector.initialize(gather_on_rank0=True)