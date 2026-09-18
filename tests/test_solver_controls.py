"""Regression checks for solver controls, without benchmark input files.

Run from the repository root: python -B -m unittest discover -s tests -v
"""
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import T_iso
from BothLandauCollision import (
    BothExplicit,
    IntraExplicit,
    ParticleSpecies,
    get_ensemble_explicit,
    get_ensemble_intra_explicit,
)


def species(count=4, mass=1.):
    result = ParticleSpecies(N=count, m=mass, n=count / 4)
    rng = np.random.default_rng(1000 + count + int(100 * mass))
    result.v = rng.normal(size=(count, 3)) / np.sqrt(mass)
    return result


def both(count1=4, count2=8, dt=.125):
    return BothExplicit(species(count1), species(count2, 5.), coulomb_log=1., dt=dt)


class SolverControlsTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(91)

    def solvers(self):
        yield IntraExplicit(species(), coulomb_log=1., dt=.125), {'N_group': 1}
        yield both(), {'N_groups': [1, 1, 1]}

    def test_isotropization_rate_matches_manuscript(self):
        self.assertAlmostEqual(
            T_iso.get_nu_iso(4., 1., 1., 1., 1.),
            0.001565397909653615,
            places=17,
        )

    def test_isotropization_script_dispatches_one_species_and_collects_errors(self):
        for fails in (False, True):
            with self.subTest(worker_fails=fails), \
                    patch('builtins.open', return_value=io.BytesIO(pickle.dumps([species()]))), \
                    patch.object(T_iso.mp, 'Pool') as pool_factory, \
                    patch.object(T_iso, 'Process_list', range(1)), \
                    patch.object(T_iso, 'N_process', 1), \
                    patch('builtins.print'):
                pool = pool_factory.return_value.__enter__.return_value
                job = pool.apply_async.return_value
                if fails:
                    job.get.side_effect = RuntimeError('worker failed')
                    with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                        T_iso.main()
                else:
                    T_iso.main()
                call = pool.apply_async.call_args.kwargs
                self.assertIs(call['func'], get_ensemble_intra_explicit)
                self.assertNotIn('s2', call['kwds'])
                self.assertEqual(call['kwds']['N_group'], 1)
                self.assertEqual(call['kwds']['special_record'], [0])
                paper_tau = 1. / 0.001565397909653615
                self.assertAlmostEqual(call['kwds']['dt'] / paper_tau, .01)
                self.assertAlmostEqual(call['kwds']['total_t'] / paper_tau, 10.)
                job.get.assert_called_once_with()

    def test_one_species_ensemble_round_trip_and_reproducibility(self):
        initial = species()
        original_v = initial.v.copy()
        # Keep temporary outputs in the workspace, not the system temp directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            kwargs = dict(dt=.125, total_t=.625, s1=initial, save_dir=directory,
                          sub_id=7, N_group=1, N_record=2, special_record=[0, 5],
                          print_status=False)
            get_ensemble_intra_explicit(**kwargs)
            with open(Path(directory) / '7.pickle', 'rb') as handle:
                saved = pickle.load(handle)
            self.assertIsInstance(saved, IntraExplicit)
            self.assertEqual(saved.step_history, [0, 2, 4, 5])
            np.testing.assert_allclose(saved.time_history, [0., .25, .5, .625])
            np.testing.assert_array_equal(saved.specific_v1[0], original_v)
            np.testing.assert_array_equal(saved.specific_v1[5], saved.s1.v)
            np.testing.assert_array_equal(initial.v, original_v)
            get_ensemble_intra_explicit(**kwargs)
            with open(Path(directory) / '7.pickle', 'rb') as handle:
                repeated = pickle.load(handle)
            np.testing.assert_array_equal(saved.s1.v, repeated.s1.v)

    def test_mixed_groups_reshuffle_only_for_enabled_operators(self):
        cases = [
            ([2, 1, 1], True, True, 6),
            ([1, 2, 2], True, True, 6),
            ([1, 1, 2], True, True, 3),
            ([2, 1, 1], False, True, 6),
            ([2, 1, 1], True, False, 0),
            ([1, 2, 2], False, True, 0),
            ([1, 1, 1], True, True, 0),
        ]
        for groups, intra, inter, expected in cases:
            with self.subTest(groups=groups, intra=intra, inter=inter):
                solver = both()
                with patch('numpy.random.shuffle', wraps=np.random.shuffle) as shuffle:
                    solver.explicit_solve(Nt=3, N_groups=groups,
                                          intra_collision=intra, inter_collision=inter)
                self.assertEqual(shuffle.call_count, expected)

    def test_histories_use_completed_steps_and_include_final_state(self):
        for count, expected in [(0, [0]), (1, [0, 1]), (8, [0, 3, 6, 8]), (9, [0, 3, 6, 9])]:
            for solver, options in self.solvers():
                with self.subTest(solver=type(solver).__name__, steps=count):
                    initial = solver.s1.v.copy()
                    solver.explicit_solve(Nt=count, N_record=3,
                                          special_record=[0, count], **options)
                    self.assertEqual(solver.step_history, expected)
                    np.testing.assert_allclose(solver.time_history, np.array(expected) * .125)
                    self.assertEqual(len(solver.T1_history), len(expected))
                    np.testing.assert_array_equal(solver.specific_v1[0], initial)
                    np.testing.assert_array_equal(solver.specific_v1[count], solver.s1.v)
                    if isinstance(solver, BothExplicit):
                        self.assertEqual(len(solver.T2_history), len(expected))
                        np.testing.assert_array_equal(solver.specific_v2[count], solver.s2.v)

    def test_continued_runs_keep_cumulative_steps_and_actual_times(self):
        for solver, options in self.solvers():
            with self.subTest(solver=type(solver).__name__):
                solver.explicit_solve(Nt=4, N_record=3, special_record=[0, 4], **options)
                snapshot = solver.specific_v1[4].copy()
                solver.dt = .25
                solver.explicit_solve(Nt=4, N_record=3, special_record=[6, 8], **options)
                self.assertEqual(solver.step_history, [0, 3, 4, 6, 8])
                np.testing.assert_allclose(solver.time_history, [0., .375, .5, 1., 1.5])
                self.assertEqual(set(solver.specific_v1), {0, 4, 6, 8})
                np.testing.assert_array_equal(solver.specific_v1[4], snapshot)

    def test_recorded_arrays_do_not_change_on_later_shuffles(self):
        for solver, _ in self.solvers():
            options = {'N_group': 2} if isinstance(solver, IntraExplicit) else {'N_groups': [2, 1, 1]}
            solver.explicit_solve(Nt=1, special_record=[1], **options)
            snapshot = solver.specific_v1[1].copy()
            first_velocity = solver.v1_1_history[1].copy()
            with patch('numpy.random.shuffle', side_effect=lambda v: v.__setitem__(slice(None), v[::-1].copy())):
                solver.explicit_solve(Nt=1, special_record=[], **options)
            np.testing.assert_array_equal(solver.specific_v1[1], snapshot)
            np.testing.assert_array_equal(solver.v1_1_history[1], first_velocity)

    def test_default_groups_are_valid_and_independent_between_solvers(self):
        for n1, n2, expected in [(4, 4, [4, 2, 2]), (6, 6, [6, 3, 3]),
                                 (6, 10, [2, 3, 5]), (5, 7, [1, 1, 1])]:
            with self.subTest(counts=(n1, n2)):
                solver = both(n1, n2)
                solver.explicit_solve(Nt=1)
                self.assertEqual(solver.N_groups, expected)
                self.assertTrue(np.isfinite(solver.s1.v).all())
                self.assertTrue(np.isfinite(solver.s2.v).all())
        for count, expected in [(7, 1), (9, 3), (8, 4)]:
            solver = IntraExplicit(species(count), coulomb_log=1., dt=.125)
            solver.explicit_solve(Nt=1)
            self.assertEqual(solver.N_group, expected)

    def test_group_arguments_are_not_mutated(self):
        for groups in ([-1, -1, -1], (-1, -1, -1), np.array([-1, -1, -1])):
            solver = both()
            solver.explicit_solve(Nt=1, N_groups=groups)
            np.testing.assert_array_equal(groups, [-1, -1, -1])

    def test_two_species_ensemble_defaults_do_not_leak_between_jobs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            for count in (4, 6):
                get_ensemble_explicit(.125, .125, species(count), species(count, 5.),
                                      directory, count, print_status=False)
                with open(Path(directory) / f'{count}.pickle', 'rb') as handle:
                    saved = pickle.load(handle)
                self.assertEqual(saved.N_groups, [count, count // 2, count // 2])

    def test_inter_only_single_particle_species_skip_intra_weights(self):
        solver = both(1, 1)
        solver.explicit_solve(Nt=2, N_groups=[-1, 0, 0], intra_collision=False)
        self.assertEqual(solver.N_groups, [1, 1, 1])
        self.assertTrue(np.isfinite(solver.s1.v).all())
        self.assertTrue(np.isfinite(solver.s2.v).all())

    def test_invalid_groups_fail_before_advancing(self):
        for groups in ([3, 1, 1], [1, 4, 1], [0, 1, 1], [-2, 1, 1],
                       [1.5, 1, 1], [True, 1, 1], [1, 1]):
            with self.subTest(groups=groups):
                solver = both()
                initial = solver.s1.v.copy()
                with self.assertRaises(ValueError):
                    solver.explicit_solve(Nt=1, N_groups=groups)
                np.testing.assert_array_equal(solver.s1.v, initial)
                self.assertEqual(solver.step, 0)
        solver = IntraExplicit(species(1), coulomb_log=1., dt=.125)
        with self.assertRaises(ValueError):
            solver.explicit_solve(Nt=1)

    def test_invalid_recording_options_fail_before_advancing(self):
        for values in ({'N_record': 0}, {'N_record': 1.5}, {'Nt': -1}, {'Nt': .5}):
            for solver, options in self.solvers():
                with self.subTest(options=values, solver=type(solver).__name__):
                    with self.assertRaises(ValueError):
                        solver.explicit_solve(**values, **options)
                    self.assertEqual(solver.step_history, [0])

    def test_energy_and_momentum_remain_conserved_with_mixed_groups(self):
        for groups in ([1, 1, 1], [2, 1, 1], [1, 2, 4]):
            with self.subTest(groups=groups):
                solver = both(8, 16, dt=.01)
                solver.explicit_solve(Nt=25, N_groups=groups)
                energy = np.array(solver.E1_total_history) + solver.E2_total_history
                momentum = np.array(solver.P1_history) + solver.P2_history
                self.assertLess(np.max(np.abs(energy / energy[0] - 1.)), 1.e-12)
                self.assertLess(np.max(np.abs(momentum - momentum[0])), 1.e-12)


if __name__ == '__main__':
    unittest.main()
