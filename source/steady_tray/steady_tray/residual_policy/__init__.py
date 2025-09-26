import gymnasium as gym

from steady_tray import agents




gym.register(
    id="G1-Residual",
    entry_point=f"{__name__}.residual_env:G1ResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_cfg:G1ResidualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualRunnerCfg",
    },
)