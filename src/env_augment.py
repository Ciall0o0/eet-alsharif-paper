"""Elevator-index permutation augmentation for symmetric-break training.

Background
----------
The PPO policy collapsed to an action-index bias: it hard-codes a preference
for obs *slots* 2/1 (elevators 2/1) regardless of physical state (proven via
permutation test on the deployed checkpoint -- swapping physical elevators 0/2
left the output unchanged). The environment is fully symmetric (no elevator id
is encoded), so a deterministic optimal policy is *invariant* to any fixed
permutation of the elevator blocks.

Fix: re-sample a fresh permutation of the elevator blocks on EVERY training
step, feed the permuted obs to the policy, then map the policy's *slot* action
back to the *physical* elevator before stepping the env. (Per-step, not
per-episode: a fixed-per-episode perm degenerates back into slot bias when an
episode is long, because each env's mapping stays constant.) Because the
buffer stores the permuted obs + slot action and the permutation changes each
step, the policy can never bind a fixed slot to a fixed physical elevator --
the optimal policy is pushed toward physical-feature reasoning.

Pure functions only (numpy), so this is safe inside both the uncompiled and
``torch.compile`` graphs (the permutation happens in the runner before the
tensor enters the policy).
"""

from __future__ import annotations

import numpy as np

# Layout of the obs (state_dim), given MAX_FLOOR=10, num_elevators=3 -> STATE_DIM=89 (A2 no-destination).
# In general: STATE_DIM = (MAX_FLOOR+3+7)*num_elevators + MAX_FLOOR*4 + 2 + 2 + 5
#            = 20*num_elevators + 40 + 9.
#   elevator blocks:  [0 .. num_elevators*20)  (each block = MAX_FLOOR+3+7 = 20)
#   up calls one-hot: [num_elevators*20 .. +10)
#   down calls:       [..+10 .. +10)
#   up dest:          [..+10 .. +10)  (zeroed in the multicall mask)
#   down dest:        [..+10 .. +10)
#   global:           [..+40 .. +9)
# Only the first `num_elevators` blocks are permuted; the call/dest/global
# blocks are order-invariant (calls are per-floor, not per-elevator).
ELEV_BLOCK = 20  # MAX_FLOOR + 3 + 7


def permute_elevator_blocks(obs: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Return a copy of ``obs`` with the elevator blocks reordered by ``perm``.

    ``perm`` is a permutation of ``range(num_elevators)`` where the output block
    at position ``k`` takes the content of the original block ``perm[k]``.
    """
    n_el = len(perm)
    out = obs.copy()
    for k in range(n_el):
        src = perm[k] * ELEV_BLOCK
        dst = k * ELEV_BLOCK
        out[dst:dst + ELEV_BLOCK] = obs[src:src + ELEV_BLOCK]
    return out


def inv_action(perm: np.ndarray, slot_action: int) -> int:
    """Map a policy *slot* action back to the physical elevator index.

    If the policy chose slot ``a`` (meaning "the elevator now sitting in block
    ``a``"), the physical elevator is ``perm[a]``. Coerce to python int so that
    numpy 0-d scalars (which some torch/tolist paths return) index cleanly.
    """
    return int(perm[int(slot_action)])


def random_perm(n_el: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random permutation of ``range(n_el)``."""
    return rng.permutation(n_el).astype(np.int64)


def _make_dummy_obs(n_el: int = 3, state_dim: int = 89) -> np.ndarray:
    obs = np.zeros(state_dim, dtype=np.float32)
    for k in range(n_el):
        obs[k * ELEV_BLOCK:(k + 1) * ELEV_BLOCK] = (k + 1) * 10.0
    obs[60:70] = 1.0
    obs[70:80] = 2.0
    obs[80:89] = 3.0
    return obs


def _self_test() -> None:
    n_el = 3
    obs = _make_dummy_obs(n_el)
    ident = np.array([0, 1, 2])
    out = permute_elevator_blocks(obs, ident)
    assert np.array_equal(out, obs), "identity perm must be a no-op"
    print("[T1] identity perm no-op: OK")
    perm = np.array([2, 0, 1])
    out = permute_elevator_blocks(obs, perm)
    assert np.allclose(out[0:ELEV_BLOCK], 30.0), "block0 expected 30"
    assert np.allclose(out[ELEV_BLOCK:2 * ELEV_BLOCK], 10.0), "block1 expected 10"
    assert np.allclose(out[2 * ELEV_BLOCK:3 * ELEV_BLOCK], 20.0), "block2 expected 20"
    assert np.allclose(out[60:70], 1.0) and np.allclose(out[70:80], 2.0), "call blocks changed!"
    assert np.allclose(out[80:89], 3.0), "global block changed!"
    print("[T2] permute block mapping + non-elevator blocks preserved: OK")
    assert inv_action(perm, 0) == 2, "slot0 -> physical 2"
    assert inv_action(perm, 1) == 0, "slot1 -> physical 0"
    assert inv_action(perm, 2) == 1, "slot2 -> physical 1"
    print("[T3] inv_action slot->physical mapping: OK")
    slot = 0
    assert inv_action(perm, slot) == perm[slot]
    print("[T4] round-trip consistency: OK")
    rng = np.random.default_rng(0)
    p1 = random_perm(n_el, rng)
    p2 = random_perm(n_el, rng)
    assert len(set(p1.tolist())) == n_el, "p1 not a permutation"
    assert len(set(p2.tolist())) == n_el, "p2 not a permutation"
    print("[T5] random_perm valid permutations: OK")
    inv = np.argsort(perm)
    restored = permute_elevator_blocks(out, inv)
    assert np.allclose(restored, obs), "inverse perm must restore original"
    print("[T6] inverse permutation restores original obs: OK")
    print("\nALL ENV_AUGMENT TESTS PASSED")


if __name__ == "__main__":
    _self_test()
