

from isaaclab.utils import configclass
from steady_tray import mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm




@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)