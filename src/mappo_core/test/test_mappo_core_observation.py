import numpy as np

from mappo_core.mappo_env import MappoEnv
from mappo_core.mappo_env import append_agent_id as env_append_agent_id
from mappo_core.mappo_obs_adapter import (
    K_MAX,
    N_ACTIONS,
    NDEF,
    RosObsBuilder,
)


RAW_OBS_DIM = 4 + K_MAX * 6 + (NDEF - 1) * 4 + 3
ACTOR_OBS_DIM = RAW_OBS_DIM + NDEF


def _fake_inputs():
    defs = [
        {
            'pos': np.array([10.0 * d, 5.0 * d]),
            'mode': 'PATROL',
            'targetID': -1,
        }
        for d in range(NDEF)
    ]
    att_pos = np.array([
        [60.0, 0.0],
        [50.0, 10.0],
        [40.0, 20.0],
        [30.0, 30.0],
        [20.0, 40.0],
        [10.0, 50.0],
    ])
    return {
        'defs': defs,
        'att_pos': att_pos,
        'est_pos': att_pos.copy(),
        'att_alive': np.ones(len(att_pos), dtype=bool),
        'att_state': ['IMMUNE'] * len(att_pos),
        'aware': np.ones((NDEF, len(att_pos)), dtype=bool),
        't_hit': np.array([50.0, 10.0, 30.0, 20.0, 40.0, 5.0]),
    }


def test_default_dimension_relationships():
    assert NDEF == 4
    assert K_MAX == 5
    assert N_ACTIONS == K_MAX + 1
    assert RAW_OBS_DIM == 49
    assert ACTOR_OBS_DIM == 53


def test_environment_produces_four_finite_actor_observations():
    env = MappoEnv(seed=7)

    raw_observations = env.reset()
    actor_observations = [
        env_append_agent_id(obs, d, env.cfg.Ndef)
        for d, obs in enumerate(raw_observations)
    ]

    assert len(raw_observations) == NDEF
    assert all(obs.shape == (RAW_OBS_DIM,) for obs in raw_observations)
    assert all(obs.shape == (ACTOR_OBS_DIM,) for obs in actor_observations)
    assert all(np.isfinite(obs).all() for obs in actor_observations)

    expected_ids = np.eye(NDEF)
    for d, obs in enumerate(actor_observations):
        np.testing.assert_array_equal(obs[-NDEF:], expected_ids[d])


def test_ros_builder_produces_four_finite_actor_observations():
    data = _fake_inputs()
    builder = RosObsBuilder(origin_xy=(0.0, 0.0), use_estimated=True)

    observations = builder.build_all(
        **data,
        t_now=0.0,
        jammer_next_avail=1.0,
        n_att_episode=len(data['att_pos']),
    )

    assert len(observations) == NDEF
    assert all(obs.shape == (ACTOR_OBS_DIM,) for obs in observations)
    assert all(np.isfinite(obs).all() for obs in observations)


def test_ros_builder_sanitizes_nonfinite_observation_values():
    data = _fake_inputs()
    data['est_pos'][0, 0] = np.nan
    data['t_hit'][1] = np.inf
    builder = RosObsBuilder(origin_xy=(0.0, 0.0), use_estimated=True)

    observations = builder.build_all(
        **data,
        t_now=0.0,
        jammer_next_avail=np.inf,
        n_att_episode=len(data['att_pos']),
    )

    assert all(np.isfinite(obs).all() for obs in observations)


def test_candidates_are_sorted_by_remaining_hit_time_and_use_one_based_ids():
    data = _fake_inputs()
    builder = RosObsBuilder(origin_xy=(0.0, 0.0), use_estimated=True)

    candidates = builder.build_candidates_only(
        **data,
        t_now=0.0,
    )

    expected = [6, 2, 4, 3, 5]
    assert len(candidates) == NDEF
    assert all(candidate_ids == expected for candidate_ids in candidates)


def test_candidates_filter_dead_unknown_and_unaware_attackers():
    data = _fake_inputs()
    data['att_state'][0] = 'UNKNOWN'
    data['att_alive'][1] = False
    data['aware'][:, 2] = False
    builder = RosObsBuilder(origin_xy=(0.0, 0.0), use_estimated=True)

    candidates = builder.build_candidates_only(
        **data,
        t_now=0.0,
    )

    expected = [6, 4, 5, -1, -1]
    assert all(candidate_ids == expected for candidate_ids in candidates)
