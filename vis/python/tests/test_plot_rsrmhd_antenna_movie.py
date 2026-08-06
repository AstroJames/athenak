"""Unit tests for fixed-scale antenna movie helpers."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_rsrmhd_antenna_movie import (  # noqa: E402
    ffmpeg_command,
    fixed_limits,
    pair_snapshots,
)


class TestAntennaMovieHelpers(unittest.TestCase):
    def test_pair_snapshots_matches_output_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output in (2, 0, 1):
                (root/f'run.prim.{output:05d}.bin').touch()
                (root/f'run.antenna.{output:05d}.bin').touch()
            pairs = pair_snapshots(root)
            self.assertEqual(
                [pair[0].name for pair in pairs],
                [f'run.prim.{output:05d}.bin' for output in range(3)],
            )
            self.assertEqual(
                [pair[1].name for pair in pairs],
                [f'run.antenna.{output:05d}.bin' for output in range(3)],
            )

    def test_pair_snapshots_rejects_missing_counterpart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run.prim.00000.bin').touch()
            with self.assertRaisesRegex(ValueError, 'matching'):
                pair_snapshots(root)

    def test_fixed_signed_limits_are_symmetric(self):
        values = [np.array([-3.0, -1.0, 0.0, 2.0])]
        lower, upper = fixed_limits(values, signed=True, percentile=100.0)
        self.assertEqual(lower, -3.0)
        self.assertEqual(upper, 3.0)

    def test_fixed_positive_limits_include_zero(self):
        values = [np.array([1.0, 2.0, 3.0])]
        lower, upper = fixed_limits(values, signed=False, percentile=100.0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 3.0)

    def test_ffmpeg_command_enforces_even_dimensions(self):
        command = ffmpeg_command(
            Path('frames/frame_%05d.png'), Path('movie.mp4'), fps=10
        )
        filter_index = command.index('-vf')
        self.assertEqual(
            command[filter_index + 1],
            'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        )


if __name__ == '__main__':
    unittest.main()
