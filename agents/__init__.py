ALGOS = {
    # Reinforcement learning algorithms
    "ppo": {"requires_demos": False, "parent": None},
    # Implementation learning algorithms
    "bc": {"requires_demos": True, "parent": None},
    "bco": {"requires_demos": True, "parent": None},
    "gail": {"requires_demos": True, "parent": None},
    "gail-bc": {"requires_demos": True, "parent": "gail"},
    "idmm": {"requires_demos": True, "parent": "gail"},
    "maad": {"requires_demos": True, "parent": "gail"},
    "tmot": {"requires_demos": True, "parent": None},
    "maad-ot": {"requires_demos": True, "parent": "tmot"},
}

ALGOS_WITH_DEMONSTRATIONS = [
    algo_name
    for algo_name, algo_options in ALGOS.items()
    if algo_options["requires_demos"]
]


def get_parent(algos: dict, algo_name: str) -> str:
    if algos[algo_name]["parent"] is None:
        return algo_name
    return algos[algo_name]["parent"]


def get_kwarg(kwargs, key):
    argv = kwargs.get(key, None)
    if argv is None:
        raise ValueError(f"{key} must be provided")
    return argv


def agent_wrapper(**kwargs):
    env = get_kwarg(kwargs, "env")
    args = get_kwarg(kwargs, "args")
    device = kwargs.get("device", "cpu")

    if args.algo in ALGOS_WITH_DEMONSTRATIONS:
        from helpers.dataset import DemoDataset

        # Create the expert demonstrations dataset from expert trajectories
        expert_dataset = DemoDataset(
            hps=args,
            env_id=args.env_id,
            expert_path=args.expert_path,
            num_demos=args.num_demos,
            env=env,
            wrap_absorb=args.wrap_absorb,
        )

    algo_parent = get_parent(ALGOS, args.algo)

    if algo_parent == "ppo":
        from agents.algos.ppo.ppo_agent import PPOAgent

        return PPOAgent(env=env, device=device, hps=args)

    elif algo_parent == "gail":
        from agents.algos.gail.gail_agent import GAILAgent

        return GAILAgent(
            env=env,
            device=device,
            hps=args,
            expert_dataset=expert_dataset,
        )

    elif algo_parent == "bc":
        from agents.algos.bc.bc_agent import BCAgent

        return BCAgent(env=env, device=device, hps=args, expert_dataset=expert_dataset)

    elif algo_parent == "bco":
        from agents.algos.bc.bco_agent import BCOAgent

        return BCOAgent(env=env, device=device, hps=args, expert_dataset=expert_dataset)

    elif algo_parent == "tmot":
        from agents.algos.tmot.tmot_agent import TMOTAgent

        return TMOTAgent(
            env=env, device=device, hps=args, expert_dataset=expert_dataset
        )

    else:
        raise ValueError(f"Unsupported algorithm {args.algo}")
