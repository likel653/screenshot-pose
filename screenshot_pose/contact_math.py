"""Reachability of independent horizontal support targets (world units)."""
import math

def pelvis_shift(legs):
    """Smallest vertical shift within every leg's maximum-reach interval.

    Each item is (hip, ankle_target, upper_length, lower_length). Empty or
    incompatible targets do not invent a root shift; the IK reports residuals.
    """
    low, high = -math.inf, math.inf
    for hip, target, upper, lower in legs:
        reach = upper + lower - 1e-6
        horizontal = (hip[0]-target[0])**2 + (hip[1]-target[1])**2
        if reach <= 0 or horizontal > reach*reach:
            return 0., False
        dz = math.sqrt(max(0., reach*reach-horizontal))
        low=max(low,target[2]-hip[2]-dz)
        high=min(high,target[2]-hip[2]+dz)
    if low > high:
        return 0., False
    return min(high,max(low,0.)), True
