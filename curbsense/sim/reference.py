"""Reference behaviors (found by CMA-ES earlier) used to sanity-check rewards and render example clips."""
from curbsense.sim.policy import NAIVE, PolicyParams

# (found by CMA-ES with 2.5 N*m hub motors)
# approaches at ~0.9 m/s, then pushes through as the front wheels drop: survives with lower shock
SAFE = PolicyParams(cruise_speed=0.90, slow_dist=0.0003, slow_speed=1.98,
                    descent_angle_deg=0.9, heading_gain=0.23, align_tol_deg=49.7)
# what RL learns from a "get across fast, don't flip" reward: launches off the curb at 2 m/s
JUMPER = PolicyParams(cruise_speed=2.0, slow_dist=0.012, slow_speed=1.99,
                      descent_angle_deg=0.5, heading_gain=2.78, align_tol_deg=64.9)

# climbing up a curb: an angled, committed approach makes it; creeping straight at it stalls on the curb face
CLIMB_OK = PolicyParams(cruise_speed=1.4, slow_dist=1.2, slow_speed=1.4, descent_angle_deg=30, heading_gain=2.5, align_tol_deg=90)
CLIMB_SLOW = PolicyParams(cruise_speed=0.4, slow_speed=0.4)

# around obstacles and people: the policy CMA-ES learned on obstacle + pedestrian twins (yields ~2 m, steers early)
AVOID = PolicyParams(cruise_speed=0.44, slow_dist=1.56, slow_speed=0.25, descent_angle_deg=6.7, heading_gain=1.25,
                     align_tol_deg=35.4, yield_dist=2.1, avoid_dist=3.56, avoid_gain=1.85)
ACTOR_NAIVE = NAIVE  # drives straight through: hits bollards, cuts close to people

BEHAVIORS = {"safe": (SAFE, "good"), "naive": (NAIVE, "bad"), "jumper": (JUMPER, "hack"),
             "climb_ok": (CLIMB_OK, "good"), "climb_slow": (CLIMB_SLOW, "bad"),
             "avoid": (AVOID, "good"), "actor_naive": (ACTOR_NAIVE, "bad")}
