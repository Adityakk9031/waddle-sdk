"""Completed motor publications stay coherent without waiting on motor IO."""

import ast
import hashlib
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
from waddle_sdk.robots import _i2rt_patches as patches


def state(value):
    return SimpleNamespace(
        pos=np.full(7, value),
        vel=np.full(7, value + 1),
        eff=np.full(7, value + 2),
        temp_mos=np.full(7, value + 3),
        temp_rotor=np.full(7, value + 4),
    )


@pytest.mark.parametrize("index", [None, 6])
def test_completed_snapshot_is_consistent_when_writer_publishes_during_read(index):
    first, second = state(1), state(10)

    class Robot:
        _gripper_index = index
        temp_record_flag = True
        accesses = 0

        @property
        def _joint_state(self):
            self.accesses += 1
            return first if self.accesses == 1 else second

        @property
        def _state_lock(self):
            raise AssertionError(
                "Feedback read must not wait for the motor transaction"
            )

    robot = Robot()
    measured = patches._observation_snapshot(robot)
    assert robot.accesses == 1
    assert np.all(measured["joint_pos"] == 1)
    assert np.all(measured["joint_vel"] == 2)
    assert np.all(measured["joint_eff"] == 3)
    assert np.all(measured["temp_mos"] == 4)
    if index is not None:
        assert measured["gripper_pos"] == [1]
        assert measured["gripper_eff"] == [3]


def test_unverified_vendor_keeps_its_original_observation_contract():
    class Custom:
        def get_observations(self):
            return {"custom": True}

    original = Custom.get_observations
    assert not patches.apply_observation_snapshot_patch(Custom)
    assert Custom.get_observations is original
    assert Custom().get_observations() == {"custom": True}


def test_verified_publisher_installs_idempotently(monkeypatch):
    class Publisher:
        def get_observations(self):
            return {}

        def _update_joint_state(self):
            pass

        def _motor_state_to_joint_state(self):
            pass

    def fingerprint(method):
        node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        return hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()

    monkeypatch.setattr(
        patches,
        "_SNAPSHOT_METHODS",
        {
            name: fingerprint(getattr(Publisher, name))
            for name in patches._SNAPSHOT_METHODS
        },
    )
    assert patches.apply_observation_snapshot_patch(Publisher)
    assert patches.apply_observation_snapshot_patch(Publisher)
    assert Publisher.get_observations is patches._observation_snapshot
