"""
Streaming-session throughput regression guard (scheduler-CPU bound).

Concurrent streaming sessions keep a large KV-resident history and extend it a
few tokens per turn. The model is truncated to a few layers and overlap
scheduling is disabled, shrinking the GPU forward so the per-step, O(resident
context) scheduler work (batch assembly, prefix match, KV bookkeeping) lands on
the critical path. This catches end-to-end regressions of those paths -- e.g.
#27965 (in-place fill_ids reconstruction).

The floor is set 3% under the measured throughput on the CI runner
(extra-b / 4-gpu-h100); see THROUGHPUT_FLOOR_TOK_S for the data and runs. A
4-GPU runner is reserved (not 1-GPU) precisely because this test is CPU-bound:
on a shared 1-GPU slice, neighbor jobs steal CPU and the throughput swings ~10%+.
"""

import concurrent.futures
import json
import random
import time
import unittest
from dataclasses import dataclass
from typing import Optional

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")

NUM_HIDDEN_LAYERS = 3
NUM_CONCURRENT = 16
CONTEXT_LEN = 30000
NUM_TURNS = 100

# Floor = 3% under the measured throughput on 4-gpu-h100: with #27965 ~2936 tok/s
# (3 runs, 2903-2966, ~2% spread), reverted ~2500 tok/s (~17% slower, caught).
# Calibration runs: gh actions runs 27536664441 / 27536667131 / 27536669466 (with),
# 27536675113 (reverted).
THROUGHPUT_FLOOR_TOK_S = 2848.0


@dataclass
class _Session:
    session_id: str
    rid: Optional[str]


def _synthetic_input_ids(
    length: int, seed: int, token_id_start: int, token_id_count: int
) -> list[int]:
    return [token_id_start + ((seed + i) % token_id_count) for i in range(length)]


def _stream_generate(
    base_url: str, input_ids: list[int], session: _Session, output_len: int
) -> int:
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
            "session_params": {"id": session.session_id, "rid": session.rid},
        },
        stream=True,
    )
    completion_tokens = 0
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        meta = json.loads(data)["meta_info"]
        completion_tokens = int(meta["completion_tokens"])
        session.rid = meta["id"]
    return completion_tokens


def bench_serving_streaming(
    base_url: str,
    *,
    num_sessions: int,
    context_len: int,
    num_turns: int,
    input_len: int = 10,
    min_gen_len: int = 1,
    max_gen_len: int = 16,
    token_id_start: int = 1000,
    token_id_count: int = 1024,
    warmup: bool = True,
) -> dict:
    def open_and_prime(session_index: int) -> _Session:
        session_id = requests.post(
            base_url + "/open_session",
            json={"capacity_of_str_len": 0, "streaming": True},
        ).json()
        session = _Session(session_id=session_id, rid=None)
        prime_ids = _synthetic_input_ids(
            context_len, session_index, token_id_start, token_id_count
        )
        _stream_generate(base_url, prime_ids, session, output_len=1)
        return session

    def run_turns(session: _Session, session_index: int) -> int:
        rng = random.Random(session_index)
        output_tokens = 0
        for turn_index in range(num_turns):
            output_len = rng.randint(min_gen_len, max_gen_len)
            input_ids = _synthetic_input_ids(
                input_len,
                session_index * num_turns + turn_index,
                token_id_start,
                token_id_count,
            )
            output_tokens += _stream_generate(base_url, input_ids, session, output_len)
        return output_tokens

    def measure() -> dict:
        requests.post(base_url + "/flush_cache")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sessions) as pool:
            sessions = list(pool.map(open_and_prime, range(num_sessions)))
            start = time.perf_counter()
            output_tokens = sum(
                pool.map(
                    lambda args: run_turns(*args),
                    [(session, idx) for idx, session in enumerate(sessions)],
                )
            )
            duration = time.perf_counter() - start
        for session in sessions:
            requests.post(
                base_url + "/close_session", json={"session_id": session.session_id}
            )
        return {
            "output_throughput": output_tokens / duration,
            "total_output_tokens": output_tokens,
            "duration_s": duration,
        }

    if warmup:
        measure()  # cold on a fresh server; discard
    return measure()


class TestStreamingSessionThroughput(StreamingSessionServerBase):
    model = "Qwen/Qwen3-0.6B"
    extra_args = [
        "--json-model-override-args",
        f'{{"num_hidden_layers": {NUM_HIDDEN_LAYERS}}}',
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "8192",
        "--schedule-policy",
        "fcfs",
        "--max-running-requests",
        "100",
        "--disable-overlap-schedule",
    ]

    def test_streaming_session_throughput(self):
        res = bench_serving_streaming(
            self.base_url,
            num_sessions=NUM_CONCURRENT,
            context_len=CONTEXT_LEN,
            num_turns=NUM_TURNS,
        )
        throughput = res["output_throughput"]
        print(
            f"\n[streaming-session throughput] sessions={NUM_CONCURRENT} "
            f"context={CONTEXT_LEN} turns={NUM_TURNS} layers={NUM_HIDDEN_LAYERS}\n"
            f"  throughput={throughput:.1f} tok/s (floor={THROUGHPUT_FLOOR_TOK_S})"
        )
        self.assertGreaterEqual(
            throughput,
            THROUGHPUT_FLOOR_TOK_S,
            f"output throughput {throughput:.1f} tok/s fell below "
            f"{THROUGHPUT_FLOOR_TOK_S}; per-step scheduler overhead may have regressed",
        )


if __name__ == "__main__":
    unittest.main()
