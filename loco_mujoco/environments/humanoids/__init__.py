from .atlas import Atlas
from .atlas_mjx import MjxAtlas
from .talos import Talos
from .talos_mjx import MjxTalos
from .unitreeH1 import UnitreeH1
from .unitreeH1_mjx import MjxUnitreeH1
from .unitreeH1v2 import UnitreeH1v2
from .unitreeH1v2_mjx import MjxUnitreeH1v2
from .unitreeG1 import UnitreeG1
from .unitreeG1_inspire import UnitreeG1Inspire
from .unitreeG1_inspire_mjx import MjxUnitreeG1Inspire
from .unitreeG1_29 import UnitreeG129
from .unitreeG1_29_mjx import MjxUnitreeG129
from .unitreeG1_percent import UnitreeG1Percent
from .unitreeG1_percent_mjx import MjxUnitreeG1Percent
from .unitreeG1_ball_inspire import UnitreeG1BallInspire
from .unitreeG1_ball_inspire_mjx import MjxUnitreeG1BallInspire
from .unitreeG1_29_ball import UnitreeG1Ball29
from .unitreeG1_29_ball_mjx import MjxUnitreeG1Ball29
from .unitreeG1_ball_percent import UnitreeG1BallPercent
from .unitreeG1_ball_percent_mjx import MjxUnitreeG1BallPercent
from .myoskeleton import MyoSkeleton
from .myoskeleton_mjx import MjxMyoSkeleton
from .unitreeG1_mjx import MjxUnitreeG1
from .apptronik_apollo import Apollo
from .apptronik_apollo_mjx import MjxApollo
from .boostert1 import BoosterT1
from .boostert1_mjx import MjxBoosterT1
from .toddlerbot import ToddlerBot
from .toddlerbot_mjx import MjxToddlerBot
from .fourier_gr1t2 import FourierGR1T2
from .fourier_gr1t2_mjx import MjxFourierGR1T2
from .skeletons import (SkeletonTorque, MjxSkeletonTorque, HumanoidTorque, SkeletonMuscle, MjxSkeletonMuscle,
                        HumanoidMuscle)


# register environments in mushroom
Atlas.register()
MjxAtlas.register()
Talos.register()
MjxTalos.register()
UnitreeH1.register()
MjxUnitreeH1.register()
UnitreeH1v2.register()
MjxUnitreeH1v2.register()
UnitreeG1.register()
UnitreeG1Inspire.register()
UnitreeG1BallInspire.register()
UnitreeG129.register()
UnitreeG1Ball29.register()
UnitreeG1Percent.register()
UnitreeG1BallPercent.register()
MjxUnitreeG1.register()
MjxUnitreeG1Inspire.register()
MjxUnitreeG1BallInspire.register()
MjxUnitreeG129.register()
MjxUnitreeG1Ball29.register()
MjxUnitreeG1Percent.register()
MjxUnitreeG1BallPercent.register()
Apollo.register()
MjxApollo.register()
BoosterT1.register()
MjxBoosterT1.register()
ToddlerBot.register()
MjxToddlerBot.register()
FourierGR1T2.register()
MjxFourierGR1T2.register()
SkeletonTorque.register()
MjxSkeletonTorque.register()
SkeletonMuscle.register()
MjxSkeletonMuscle.register()
MyoSkeleton.register()
MjxMyoSkeleton.register()

# compatability with old names
HumanoidTorque.register()
HumanoidMuscle.register()

from gymnasium import register

# register gymnasium wrapper environment
register("LocoMujoco",
         entry_point="loco_mujoco.core.wrappers.gymnasium:GymnasiumWrapper"
         )
