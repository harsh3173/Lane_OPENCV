"""raypilot — classical, image-only ray-cast driving pilot (no NN, no CTE/reward).

Public surface:
    from raypilot import RayPilot, draw, RecoveryController, RayPilotPart
    from raypilot.ray_mask import cast_rays, calibrate, seed_ref, list_imgs, numeric_key
"""
from .ray_mask import cast_rays, calibrate, seed_ref, list_imgs, numeric_key
from .pilot import RayPilot, draw
from .recovery import RecoveryController, DRIVE, SLOW, STOP, REVERSE, STUCK
from .donkey_part import RayPilotPart
from .flow_field import FlowField, FlowPart

__all__ = [
    "RayPilot", "draw", "RecoveryController", "RayPilotPart", "FlowField", "FlowPart",
    "cast_rays", "calibrate", "seed_ref", "list_imgs", "numeric_key",
    "DRIVE", "SLOW", "STOP", "REVERSE", "STUCK",
]
