from dataclasses import dataclass
import random
from .api import Step, StepRow


@dataclass(frozen=True)
class ScheduledStep:
    step: Step
    started: tuple
    finished: tuple


def select_requests(dataset, config):
    requests = list(dataset.spec.requests)
    if config.order == "shuffled": random.Random(config.seed).shuffle(requests)
    requests = requests[config.request_offset:]
    return tuple(requests[:config.max_requests] if config.max_requests else requests)


class DecodeScheduler:
    """FIFO continuous batching; a slot is replaced only at the next step."""
    def steps(self, requests, config, start_step=0):
        active, cursor, step_id = [], 0, start_step
        while cursor < len(requests) or active:
            started = []
            while len(active) < config.batch_size and cursor < len(requests):
                request = requests[cursor]
                active.append([cursor, request, 0])
                started.append(request)
                cursor += 1
            step = Step(step_id, tuple(StepRow(r.request_id, t, i) for i, r, t in active))
            finished = tuple(r for _, r, t in active if t + 1 == min(r.token_count, config.max_tokens or r.token_count))
            yield ScheduledStep(step, tuple(started), finished)
            done = {r.request_id for r in finished}
            active = [[i, r, t + 1] for i, r, t in active if r.request_id not in done]
            step_id += 1
