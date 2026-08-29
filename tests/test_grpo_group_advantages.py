"""Pinned veRL GRPO grouping: shared uid vs per-row uid.

Uses ``compute_grpo_outcome_advantage`` from the Stage-1 veRL checkout.
Size-1 groups are NOT zeroed: mean is forced to 0 and std to 1, so the
advantage equals the raw reward. Shared-uid mixed groups are centered.
"""

from __future__ import annotations

import numpy as np
import torch

from budget_coder_rl.eval.m4a import scalar_advantage
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _outcome_batch(rewards: list[float], mask_lengths: list[int], width: int = 6):
    bsz = len(rewards)
    token_rewards = torch.zeros(bsz, width, dtype=torch.float32)
    response_mask = torch.zeros(bsz, width, dtype=torch.float32)
    for i, (reward, length) in enumerate(zip(rewards, mask_lengths)):
        response_mask[i, :length] = 1.0
        token_rewards[i, length - 1] = float(reward)
    return token_rewards, response_mask


def _scalars(advantages: torch.Tensor, mask: torch.Tensor) -> list[float]:
    out: list[float] = []
    for i in range(advantages.size(0)):
        out.append(
            scalar_advantage(advantages[i].tolist(), mask[i].tolist())
        )
    return out


def test_shared_uid_mixed_rewards_nonzero_relative_advantages():
    rewards = [0.0, 0.5, 0.0, 0.5]
    token_rewards, mask = _outcome_batch(rewards, [2, 3, 4, 2])
    uid = np.array(["same-task"] * 4, dtype=object)
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=uid,
    )
    assert torch.equal(advantages, returns)
    scalars = _scalars(advantages, mask)
    assert any(abs(value) > 1e-6 for value in scalars)
    assert min(scalars) < 0 < max(scalars)
    assert abs(sum(scalars)) < 1e-5
    for scalar, reward in zip(scalars, rewards):
        assert abs(scalar - reward) > 0.1
    assert torch.all(advantages[mask == 0] == 0)
    assert torch.all(advantages[mask == 1] != 0)


def test_distinct_uids_do_not_center_against_siblings():
    rewards = [0.0, 0.5, 0.0, 0.5]
    token_rewards, mask = _outcome_batch(rewards, [2, 3, 4, 2])
    uid = np.array(["a", "b", "c", "d"], dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=uid,
    )
    scalars = _scalars(advantages, mask)
    for scalar, reward in zip(scalars, rewards):
        assert abs(scalar - reward) < 1e-5
    assert abs(sum(scalars)) > 0.4


def test_singleton_nonzero_reward_is_not_zeroed():
    token_rewards, mask = _outcome_batch([0.5], [3])
    uid = np.array(["alone"], dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=uid,
    )
    assert abs(_scalars(advantages, mask)[0] - 0.5) < 1e-5
