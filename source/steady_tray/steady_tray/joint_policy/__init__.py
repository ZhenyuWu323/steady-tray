import gymnasium as gym

from steady_tray import agents



""" G1 Residual Locomanipulation Whole Body"""

gym.register(
    id="G1-Joint-Locomotion",
    entry_point=f"{__name__}.joint_locomotion_env:G1JointLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_locomotion_cfg:G1JointLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JointRunnerCfg",
    },
)