"""Big-O and FLOP tracers. Numbers are computed, never guessed in the paper."""

from __future__ import annotations

from dataclasses import dataclass

from uaere.types import HOP_LENGTH, N_FFT, N_MELS, WINDOW_SECONDS, WORKING_FS


@dataclass(frozen=True)
class Complexity:
    name: str
    big_o: str
    flops: float
    peak_ram_bytes: float


def l0_rfft(n: int | None = None) -> Complexity:
    n = int(n or WORKING_FS * WINDOW_SECONDS)
    # rFFT ~ 5 N log2 N real flops (Bluestein/split-radix bound)
    import math

    flops = 5.0 * n * math.log2(max(n, 2))
    return Complexity("L0 rFFT + moments", "O(N log N)", flops, 16.0 * n)


def l1_mel(n: int | None = None) -> Complexity:
    n = int(n or WORKING_FS * WINDOW_SECONDS)
    n_frames = 1 + (n - N_FFT) // HOP_LENGTH
    flops = float(n_frames * N_FFT * N_MELS)  # gemm of |STFT| against Mel filterbank
    ram = n_frames * N_MELS * 4.0
    return Complexity("L1 Mel + IN", "O(N_frames · N_fft · B)", flops, ram)


def kg_bfs(n_v: int, n_e: int) -> Complexity:
    return Complexity("L3 KG BFS + SCM", "O(|V|+|E|)", float(n_v + n_e), 64.0 * (n_v + n_e))
