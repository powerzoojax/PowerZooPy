"""Backward-compatibility tests: verify existing make_task_env("name") still works."""

import pytest


class TestBackwardCompatibility:
    """Ensure existing public API is unchanged after rl/ module additions."""

    def test_make_task_env_str_unchanged(self):
        from powerzoo.tasks import make_task_env
        env = make_task_env('battery_arbitrage', split='train')
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_make_task_str_unchanged(self):
        from powerzoo.tasks import make_task
        task = make_task('battery_arbitrage')
        assert task.agent_mode == 'single'

    def test_list_tasks_unchanged(self):
        from powerzoo.tasks import list_tasks
        tasks = list_tasks()
        assert isinstance(tasks, list)
        assert 'battery_arbitrage' in tasks

    def test_make_task_env_kwargs_unchanged(self):
        from powerzoo.tasks import make_task_env
        env = make_task_env('battery_arbitrage', split='test')
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_make_task_env_pettingzoo_unchanged(self):
        try:
            import pettingzoo  # noqa: F401
        except ImportError:
            pytest.skip("pettingzoo not installed")
        from powerzoo.tasks import make_task_env
        env = make_task_env('marl_opf', split='train', framework='pettingzoo')
        obs_dict, _ = env.reset(seed=0)
        assert isinstance(obs_dict, dict)
        env.close()

    def test_make_task_env_dict_is_new_feature(self):
        """Dict path in make_task_env is a new addition, verify it works."""
        import warnings
        from powerzoo.tasks import make_task_env
        config = {
            'grid': {'type': 'distribution', 'case': 'case33bw'},
            'resources': [
                {'type': 'battery', 'bus_id': 6, 'capacity_mwh': 1.0, 'charge_power_kw': 500}
            ],
            'episode': {'max_steps': 24},
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            env = make_task_env(config)
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_top_level_make_env_exposed(self):
        """make_env is now exported from the top-level powerzoo package."""
        import powerzoo
        assert hasattr(powerzoo, 'make_env')

    def test_top_level_rlconfig_exposed(self):
        import powerzoo
        assert hasattr(powerzoo, 'RLConfig')
