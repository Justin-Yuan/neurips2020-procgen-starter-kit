from typing import List, Tuple
import time

from ray.util.iter import from_actors, LocalIterator
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import GradientType, SampleBatchType, \
    STEPS_SAMPLED_COUNTER, LEARNER_INFO, SAMPLE_TIMER, \
    GRAD_WAIT_TIMER, _check_sample_batch_type
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch

PolicyID = 'str'

class SelectExperiences:
    """Callable used to select experiences from a MultiAgentBatch.
    This should be used with the .for_each() operator.
    Examples:
        >>> rollouts = ParallelRollouts(...)
        >>> rollouts = rollouts.for_each(SelectExperiences(["pol1", "pol2"]))
        >>> print(next(rollouts).policy_batches.keys())
        {"pol1", "pol2"}
    """

    def __init__(self, policy_ids: List[PolicyID]):
        assert isinstance(policy_ids, list), policy_ids
        self.policy_ids = policy_ids

    def __call__(self, samples: SampleBatchType) -> SampleBatchType:
        _check_sample_batch_type(samples)

        if isinstance(samples, MultiAgentBatch):
            samples = MultiAgentBatch({
                k: v
                for k, v in samples.policy_batches.items()
                if k in self.policy_ids
            }, samples.count)

        return samples

class StandardizeFields:
    """Callable used to standardize fields of batches.
    This should be used with the .for_each() operator. Note that the input
    may be mutated by this operator for efficiency.
    Examples:
        >>> rollouts = ParallelRollouts(...)
        >>> rollouts = rollouts.for_each(StandardizeFields(["advantages"]))
        >>> print(np.std(next(rollouts)["advantages"]))
        1.0
    """

    def __init__(self, fields: List[str]):
        self.fields = fields

    def __call__(self, samples: SampleBatchType) -> SampleBatchType:
        _check_sample_batch_type(samples)
        wrapped = False

        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({
                DEFAULT_POLICY_ID: samples
            }, samples.count)
            wrapped = True

        for policy_id in samples.policy_batches:
            batch = samples.policy_batches[policy_id]
            for field in self.fields:
                batch[field] = standardized(batch[field])

        if wrapped:
            samples = samples.policy_batches[DEFAULT_POLICY_ID]

        return samples