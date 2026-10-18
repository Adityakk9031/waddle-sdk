"""Aligned RGB-D remains paired, with scoped errors for missing filter output."""

from types import SimpleNamespace

import numpy as np
import pytest
from waddle_sdk.cameras.orbbec import OrbbecDriver


@pytest.mark.parametrize("available", [False, True])
def test_aligned_orbbec_samples_are_paired_or_report_the_alignment_gap(available):
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.arange(6, dtype=np.uint16).reshape(2, 3)

    def frame(data):
        return SimpleNamespace(
            get_width=lambda: 3,
            get_height=lambda: 2,
            get_data=lambda: data,
            get_depth_scale=lambda: 1.0,
        )

    aligned = SimpleNamespace(
        get_color_frame=lambda: frame(rgb), get_depth_frame=lambda: frame(depth)
    )
    driver = object.__new__(OrbbecDriver)
    driver._pipeline = SimpleNamespace(wait_for_frames=lambda _timeout: object())
    driver._align = SimpleNamespace(
        process=lambda _frames: aligned if available else None
    )
    if not available:
        with pytest.raises(
            RuntimeError, match="Orbbec alignment returned no RGB-D frameset"
        ):
            driver.capture()
        return
    sample = driver.capture()
    np.testing.assert_array_equal(sample.rgb, rgb)
    np.testing.assert_array_equal(sample.depth, depth)
    assert driver.depth_scale_mm == 1.0
