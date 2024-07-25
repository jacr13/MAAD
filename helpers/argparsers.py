import argparse

from agents import ALGOS
from helpers.misc_util import boolean_flag


def argparser(description="Milozero Experiment"):
    """Create an argparse.ArgumentParser"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Primary
    parser.add_argument("--env_id", help="environment identifier", default="Hopper-v2")
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--expert_path", help="demos location", type=str, default=None)

    # wandb
    parser.add_argument("--wandb_project", help="wandb project name", default="DEFAULT")
    parser.add_argument("--wandb_entity", help="wandb entity name", default=None)
    parser.add_argument("--wandb_mode", help="wandb mode", default="online")
    parser.add_argument(
        "--wandb_log_freq", help="wandb log frequency (watch)", type=int, default=1
    )
    parser.add_argument(
        "--wandb_run_path",
        help='wandb run path in the form "<entity>/<project>/<run_id>"',
        default=None,
    )
    boolean_flag(parser, "wandb_download_replace", default=False)

    # Generic
    parser.add_argument("--uuid", type=str, default=None)
    boolean_flag(parser, "cuda", default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoints")
    parser.add_argument("--log_dir", type=str, default="data/logs")
    boolean_flag(parser, "render", help="render the interaction traces", default=False)
    parser.add_argument("--video_dir", type=str, default="data/videos")
    boolean_flag(parser, "record", help="record the interaction traces", default=False)
    parser.add_argument("--record_frequency", type=int, default=100)
    parser.add_argument("--record_n_ep", type=int, default=2)
    parser.add_argument("--record_format", type=str, default="mp4")
    parser.add_argument("--task", type=str, choices=["train", "eval"], default=None)
    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGOS.keys(),
        default=None,
    )
    boolean_flag(parser, "trainer_worker", default=False)
    parser.add_argument(
        "--execution_time",
        help="time allowed in d-HH:MM:SS format",
        type=str,
        default=None,
    )

    # Training
    parser.add_argument(
        "--save_frequency",
        help="save model every xx iterations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_timesteps",
        help="total number of interactions",
        type=int,
        default=int(1e10),
    )
    parser.add_argument("--training_num_envs", type=int, default=64)
    parser.add_argument("--evaluation_num_envs", type=int, default=10)
    parser.add_argument("--training_steps_per_iter", type=int, default=64)
    parser.add_argument("--eval_steps_per_iter", type=int, default=10)
    parser.add_argument("--eval_frequency", type=int, default=10)

    # Pretraining
    boolean_flag(parser, "pretrain", default=False)
    parser.add_argument("--pretrain_epochs", type=int, default=200)

    # Model
    parser.add_argument("--p_perception_stack", type=str, default=None)
    parser.add_argument("--v_perception_stack", type=str, default=None)
    parser.add_argument("--d_perception_stack", type=str, default=None)
    boolean_flag(parser, "layer_norm", default=False)
    boolean_flag(parser, "shared_value", default=False)
    boolean_flag(parser, "shared_value_policy_detached", default=False)
    boolean_flag(
        parser,
        "batch_norm_all",
        default=False,
    )  # set all *_batch_norm params to the same
    boolean_flag(parser, "p_batch_norm", default=False)
    boolean_flag(parser, "v_batch_norm", default=False)
    parser.add_argument("--policy_std_init", type=float, default=0.01)

    # Optimization
    parser.add_argument("--p_lr", type=float, default=3e-4)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["linear", "constant", "adaptive", "step"],
        default="constant",
    )
    parser.add_argument(
        "--wd_scale",
        help="weight decay scale",
        type=float,
        default=0.001,
    )

    # Algorithm
    parser.add_argument(
        "--rollout_len",
        help="number of interactions per iteration",
        type=int,
        default=2048,
    )
    parser.add_argument("--optim_epochs_per_iter", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", help="discount factor", type=float, default=0.99)
    parser.add_argument("--mem_size", type=int, default=int(1e5))
    parser.add_argument(
        "--noise_type",
        type=str,
        default="adaptive-param_0.2, ou_0.1, normal_0.1",
    )
    parser.add_argument("--pn_adapt_frequency", type=float, default=50)
    parser.add_argument(
        "--polyak",
        type=float,
        default=0.005,
        help="soft target nets update",
    )
    parser.add_argument(
        "--targ_up_freq",
        type=int,
        default=100,
        help="hard target nets update",
    )
    boolean_flag(parser, "n_step_returns", default=False)
    parser.add_argument("--lookahead", help="num lookahead steps", type=int, default=10)
    boolean_flag(parser, "ret_norm", default=False)
    boolean_flag(parser, "popart", default=False)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument(
        "--eps",
        help="ppo ratio clipping for surrogate loss",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--kl_weight",
        help="weight for the KL loss",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--value_clip",
        help="ration clipping for the value function",
        type=float,
        default=1,
    )
    parser.add_argument("--baseline_scale", type=float, default=0.5)
    parser.add_argument(
        "--p_ent_reg_scale",
        help="p entropy reg coeff",
        type=float,
        default=0.0,
    )
    # MINE
    parser.add_argument(
        "--p_mi_reg_scale",
        help="p mutual information reg coeff",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--mine_version",
        type=str,
        help="MINE version [MINE, MINEf, DRT]",
        default="MINE",
    )
    parser.add_argument(
        "--mine_hidsize",
        type=int,
        help="MINE hidden size",
        default=128,
    )
    # Inverse Dynamic Model
    parser.add_argument(
        "--idm_model",
        type=str,
        help="Type of Inverse Dynamic Model (default: mlp)",
        default="mlp",
    )
    parser.add_argument(
        "--idm_hidsize",
        type=int,
        help="Inverse Dynamic Model hidden size",
        default=128,
    )
    parser.add_argument("--idm_lr", type=float, default=3e-4)
    parser.add_argument("--idm_loss", type=str, default="mse")
    boolean_flag(parser, "idm_pretrained", default=False)
    boolean_flag(parser, "idm_pretrained_from_wandb", default=False)
    parser.add_argument("--idm_pretrained_path", type=str, default=None)

    parser.add_argument(
        "--idm_replaybuffer_type",
        type=str,
        default=None,
        choices=[
            "normal",
            "prioritized",
        ],
    )
    parser.add_argument(
        "--idm_mdn_n_gaussians",
        type=int,
        help="Number of gaussians we should learn in MDN inverse dynamic model",
        default=3,
    )

    # Prioritized replay
    boolean_flag(parser, "prioritized_replay", default=False)
    parser.add_argument("--alpha", help="how much prioritized", type=float, default=0.3)
    parser.add_argument(
        "--beta",
        help="importance weights usage",
        type=float,
        default=1.0,
    )
    boolean_flag(parser, "ranked", default=False)
    boolean_flag(parser, "unreal", default=False)

    # Adversarial imitation
    parser.add_argument("--g_steps", type=int, default=3)
    parser.add_argument("--d_steps", type=int, default=1)
    parser.add_argument("--d_lr", type=float, default=3e-4)
    boolean_flag(parser, "state_only", default=False)
    boolean_flag(parser, "state_state", default=False)
    boolean_flag(parser, "minimax_only", default=True)
    parser.add_argument("--ent_reg_scale", type=float, default=0.001)
    parser.add_argument(
        "--d_ent_reg_scale",
        help="d entropy reg coeff",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--num_demos",
        help="number of expert demo trajs for imitation",
        type=int,
        default=None,
    )
    boolean_flag(
        parser,
        "expert_demo_subsample",
        help="subsample expert demo trajs for imitation",
        default=True,
    )
    parser.add_argument(
        "--expert_demo_subsample_rate",
        help="subsample rate for expert demo trajs",
        type=int,
        default=20,
    )
    boolean_flag(
        parser,
        "spectral_norm",
        help="whether to use spectral norm",
        default=True,
    )
    boolean_flag(
        parser,
        "grad_pen",
        help="whether to use gradient penalty",
        default=True,
    )
    parser.add_argument("--grad_pen_targ", type=float, default=1.0)
    parser.add_argument("--grad_pen_scale", type=float, default=10.0)
    parser.add_argument(
        "--grad_pen_type",
        type=str,
        default=None,
        choices=["wgan", "hyperwgan", "dragan", "nagard"],
    )
    boolean_flag(
        parser,
        "one_sided_pen",
        help="whether to use the one-sided version",
        default=True,
    )
    boolean_flag(parser, "historical_patching", default=True)
    parser.add_argument("--fake_ls_type", type=str, default="none")
    parser.add_argument("--real_ls_type", type=str, default="random-uniform_0.7_1.2")
    parser.add_argument("--syn_rew_scale", type=float, default=1.0)
    boolean_flag(parser, "wrap_absorb", default=False)
    boolean_flag(parser, "d_batch_norm", default=False)

    boolean_flag(parser, "kye_p", default=False)
    parser.add_argument("--kye_p_scale", type=float, default=0.1)
    boolean_flag(parser, "kye_mixing", default=True)
    boolean_flag(parser, "adaptive_aux_scaling", default=False)

    parser.add_argument(
        "--reward_type",
        type=str,
        default=None,
        choices=[
            "gail",
            "red",
            "gail_red_mod",
            "gail_kye_mod",
            "gail_dyn_mod",
            "gail_grad_mod",
            "gail_red_grad_mod",
        ],
    )

    # BC
    parser.add_argument("--bc_epochs", type=int, default=200)
    parser.add_argument("--bc_patience", type=int, default=10)
    parser.add_argument("--bco_post_rollout_len", type=int, default=5000)

    # GAIL-BC
    parser.add_argument("--gail_bc_lambda", type=float, default=1.0)
    parser.add_argument("--gail_bc_lambda_decay", type=float, default=0.99)
    parser.add_argument("--gail_bc_loss", type=str, default="mse")

    # Evaluation / Resume training
    boolean_flag(parser, "resume_training", default=False)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--num_trajs",
        help="number of trajectories to evaluate",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--iter_num",
        help="iteration to evaluate/resume the model or best to load best model",
        default=None,
    )
    boolean_flag(parser, "save_traj", default=False)
    boolean_flag(parser, "save_traj_pixels", default=False)
    parser.add_argument("--save_traj_dir", type=str, default="data/demos")
    boolean_flag(parser, "compare_plots", default=False)

    # TODO: verify if there is a difference between 10(ppo/gail) and 40 (ddpg/same)
    parser.add_argument(
        "--maxlen",
        help="maximum length for the eval deque",
        type=int,
        default=10,
    )

    # ROT
    parser.add_argument("--rot_update_target_every", type=int, default=10000)
    parser.add_argument("--rot_sinkhorn_rew_scale", type=int, default=200)
    parser.add_argument(
        "--ot_rewards",
        type=str,
        default="euclidean",
        choices=[
            "sinkhorn_cosine",
            "sinkhorn_euclidean",
            "cosine",
            "euclidean",
            "pwil",
        ],
    )
    boolean_flag(parser, "rot_auto_rew_scale", default=True)
    parser.add_argument("--rot_auto_rew_scale_factor", type=int, default=100)

    return parser
