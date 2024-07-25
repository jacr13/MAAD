# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MuJoCo environments.

MUJOCO_ROBOTS = [
    "InvertedPendulum",
    "InvertedDoublePendulum",
    "Reacher",
    "Hopper",
    "HalfCheetah",
    "Walker2d",
    "Ant",
    "Humanoid",
]

MYO_SUITE = [
    "myoChallengeBaodingP1-v1",
    "myoChallengeBaodingP2-v1",
    "myoChallengeDieReorientP1-v0",
]

MUJOCO_ENVS = ["{}-v2".format(name) for name in MUJOCO_ROBOTS]
MUJOCO_ENVS.extend(["{}-v3".format(name) for name in MUJOCO_ROBOTS])
MUJOCO_ENVS.extend(["{}-v4".format(name) for name in MUJOCO_ROBOTS])
MUJOCO_ENVS.extend(MYO_SUITE)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DM Control environments.

DMC_ROBOTS = [
    "Acrobot-Swingup",
    "Cartpole-Swingup",
    "Cartpole-Two_poles",
    "Cartpole-Three_poles",
    "Hopper-Hop",
    "Cheetah-Run",
    "Walker-Walk",
    "Walker-Run",
    "Stacker-Stack_2",
    "Stacker-Stack_4",
    "Humanoid-Walk",
    "Humanoid-Run",
    "Humanoid-Run_Pure_State",
    "Humanoid_CMU-Stand",
    "Humanoid_CMU-Run",
    "Quadruped-Walk",
    "Quadruped-Run",
    "Quadruped-Escape",
    "Quadruped-Fetch",
    "Dog-Run",
    "Dog-Fetch",
]


DMC_ENVS = ["{}-Feat-v0".format(name) for name in DMC_ROBOTS]
DMC_ENVS.extend(["{}-Pixels-v0".format(name) for name in DMC_ROBOTS])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DM Control environments with envpool.

DMC_EV_ROBOTS = [
    "Acrobot-Swingup",
    "Cartpole-Swingup",
    "Cartpole-Two_poles",
    "Cartpole-Three_poles",
    "Hopper-Hop",
    "Cheetah-Run",
    "Walker-Walk",
    "Walker-Run",
    "Humanoid-Walk",
    "Humanoid-Run",
    "Humanoid-Run_Pure_State",
    "Humanoid_CMU-Stand",
    "Humanoid_CMU-Run",
]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> benchmarks.

BENCHMARKS = {
    "mujoco": MUJOCO_ENVS,
    "dmc": DMC_ENVS,
}
