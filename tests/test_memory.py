import pytest
import torch

from boundary_em.memory import compose_key, differentiable_read
from boundary_em.task import TaskConfig, generate_episode


def test_memory_read_is_differentiable():
    query = torch.tensor([0.8, 0.2], requires_grad=True)
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    values = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    readout = differentiable_read(query, keys, values, temperature=0.2)
    readout.value[1].backward()

    assert readout.attention.sum().item() == pytest.approx(1.0)
    assert query.grad is not None and query.grad.abs().sum().item() > 0
    assert keys.grad is not None and keys.grad.abs().sum().item() > 0


def test_memory_read_is_differentiable_with_respect_to_encoding_strength() -> None:
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    encoding_strengths = torch.tensor([0.8, 0.2], requires_grad=True)

    readout = differentiable_read(
        query,
        keys,
        values,
        temperature=0.2,
        encoding_strengths=encoding_strengths,
    )
    readout.value[1].backward()

    assert readout.attention.tolist() == pytest.approx([0.8, 0.2])
    assert encoding_strengths.grad is not None
    assert encoding_strengths.grad[0] < 0
    assert encoding_strengths.grad[1] > 0


def test_incomplete_trace_competes_with_the_endpoint_trace():
    config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    episode = generate_episode(config, seed=29)
    keys = torch.stack(
        [
            compose_key(episode.cue, state, mask)
            for state, mask in zip(episode.states, episode.masks, strict=True)
        ]
    )
    query = compose_key(episode.cue, episode.query_state, episode.query_mask)
    endpoint = differentiable_read(
        query, keys[-1:], episode.states[-1:], temperature=0.1
    )
    mixed = differentiable_read(
        query, keys[[1, 3]], episode.states[[1, 3]], temperature=0.1
    )
    held_out = episode.query_mask == 0

    endpoint_mse = torch.mean(
        (endpoint.value[held_out] - episode.features[held_out]) ** 2
    )
    mixed_mse = torch.mean((mixed.value[held_out] - episode.features[held_out]) ** 2)

    assert endpoint_mse.item() == pytest.approx(0.0)
    assert mixed_mse.item() > 0.5
    assert mixed.attention[0] > mixed.attention[1]
