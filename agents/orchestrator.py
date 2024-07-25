import glob
import os
import os.path as osp
import shutil
import time
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
import wandb

from helpers import logger
from helpers.console_util import log_iter_info, timed_cm_wrapper
from helpers.dataset import save_dict_h5py
from helpers.distributed_util import mpi_save_random_states
from helpers.env_makers import get_benchmark
from helpers.opencv_util import record_video
from helpers.timeout_handler import TimeoutHandler

MAXLEN = 40


RECORD = False


def ep_generator(env, agent, render):  # noqa: C901
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """
    global RECORD

    benchmark = get_benchmark(agent.hps.env_id)

    if RECORD:
        # Note: leave structure intact for extensibility to other envs

        def bgr_to_rgb(x):
            _b = np.expand_dims(x[..., 0], -1)
            _g = np.expand_dims(x[..., 1], -1)
            _r = np.expand_dims(x[..., 2], -1)
            rgb_x = np.concatenate([_r, _g, _b], axis=-1)
            del x, _b, _g, _r
            return rgb_x

        kwargs = {"mode": "rgb_array"}
        if benchmark in ["metaworld"]:
            kwargs = {}
        if benchmark in ["mujoco", "dmc", "metaworld"]:

            def _render():
                if benchmark == "metaworld":
                    return env.render(**kwargs)
                return bgr_to_rgb(env.render(**kwargs))

        else:
            raise ValueError("unsupported benchmark")

    rst = env.reset()
    if isinstance(rst, tuple) and len(rst) == 2:
        ob, _ = rst
    else:
        ob = rst
    ob = np.array(ob)

    if RECORD:
        ob_orig = _render()

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs0 = []
    obs1 = []
    dones1 = []
    if RECORD:
        obs_render = []

    acs = []
    vs = []
    env_rews = []

    while True:
        ac, v, _ = agent.predict(ob, True)

        if not agent.is_discrete:
            # NaN-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, env.action_space.low, env.action_space.high)
        else:
            ac = ac if isinstance(ac, int) else ac.item()

        obs0.append(ob)
        if RECORD:
            obs_render.append(ob_orig)
        acs.append(ac)
        vs.append(v)
        step = env.step(ac)
        if len(step) == 4:
            new_ob, env_rew, done, info = step
        elif len(step) == 5:
            new_ob, env_rew, terminated, truncated, info = step
            done = terminated or truncated
        else:
            raise ValueError(f"unexpected step length: {len(step)}")

        obs1.append(new_ob)
        dones1.append(done)

        if render:
            if agent.hps.env_id.startswith("myo"):
                env.sim.render(mode="window")
            else:
                env.render()

        if RECORD:
            ob_orig = _render()

        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = np.array(deepcopy(new_ob))
        if np.array(done).all():
            obs0 = np.array(obs0)
            obs1 = np.array(obs1)
            dones1 = np.array(dones1)
            if RECORD:
                obs_render = np.array(obs_render)
            acs = np.array(acs)
            env_rews = np.array(env_rews)

            out = {
                "obs0": obs0,
                "obs1": obs1,
                "dones1": np.array(dones1),
                "acs": acs,
                "env_rews": env_rews,
                "ep_lens": cur_ep_len,
                "ep_env_rets": cur_ep_env_ret,
            }
            if vs[-1] is not None:
                out["vs"] = vs

            if RECORD:
                out.update({"obs_render": obs_render})
            if benchmark == "dmc_ep":
                for k, v in out.items():
                    if k in [
                        "obs0",
                        "obs1",
                        "acs",
                        "vs",
                        "logps",
                        "env_rews",
                        "syn_rews",
                        "dones",
                    ]:
                        out[k] = np.array(out[k]).swapaxes(0, 1)
                    else:
                        out[k] = np.array(out[k])
            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs0 = []
            obs1 = []
            dones1 = []
            if RECORD:
                obs_render = []
            acs = []
            vs = []
            env_rews = []

            rst = env.reset()
            if isinstance(rst, tuple) and len(rst) == 2:
                ob, _ = rst
            else:
                ob = rst
            ob = np.array(ob)

            if RECORD:
                ob_orig = _render()


def evaluate(args, env, agent, experiment_name):
    benchmark = get_benchmark(agent.hps.env_id)
    # Create episode generator
    global RECORD
    RECORD = args.record
    ep_gen = ep_generator(
        env,
        agent,
        args.render,
    )

    if args.record:
        vid_dir = osp.join(args.video_dir, experiment_name)
        os.makedirs(vid_dir, exist_ok=True)

    # Load the model
    if args.iter_num == "best":
        args.model_path = osp.join(args.model_path, "model_best")
    agent.load(args.model_path, args.iter_num)
    logger.info("model loaded from path:\n  {}".format(args.model_path))
    logger.info("model loaded from iteration number: {}".format(args.iter_num))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(args.num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, args.num_trajs))
        agent.to("cpu")
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj["ep_lens"], traj["ep_env_rets"]
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)

        # log some information about episode
        logger.info(f"\tEpisode length: {ep_len}")
        logger.info(f"\tEpisode return: {ep_env_ret}")

        if args.record:
            benchmark = get_benchmark(agent.hps.env_id)
            assert benchmark in [
                "mujoco",
                "dmc",
                "metaworld",
            ], "record is not supported for this environment"
            # Record a video of the episode
            logger.info("recording episode @: {}".format(vid_dir))
            record_video(
                vid_dir,
                f"eval_{i}",
                traj["obs_render"],
                record_format=args.record_format,
            )

        if args.save_traj:
            traj.pop("vs", None)
            if not args.save_traj_pixels:
                traj.pop("obs_render", None)
            path = osp.join(args.save_traj_dir, args.env_id, args.uuid)
            os.makedirs(path, exist_ok=True)
            save_dict_h5py(traj, osp.join(path, f"{args.env_id}_demo{i:03}.h5"))

    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)

    # log env len statistics
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_len_std", np.std(ep_lens))
    logger.record_tabular("ep_len_min", np.min(ep_lens))
    logger.record_tabular("ep_len_max", np.max(ep_lens))

    logger.record_tabular("ep_env_ret_mean", int(ep_env_ret_mean))
    logger.record_tabular("ep_env_ret_std", int(np.std(ep_env_rets)))
    logger.record_tabular("ep_env_ret_min", int(np.min(ep_env_rets)))
    logger.record_tabular("ep_env_ret_max", int(np.max(ep_env_rets)))

    logger.dump_tabular()


def learn(args, rank, env, eval_env, agent, experiment_name):  # noqa: C901
    benchmark = get_benchmark(agent.hps.env_id)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger)

    num_iters = args.num_timesteps // args.rollout_len
    iters_so_far = 0
    timesteps_so_far = 0
    best_eval_env_ret = np.NINF
    tstart = args.get("__start_time", time.time())

    if args.resume_training:
        assert isinstance(args.iter_num, int), "iter_num must be an integer"
        iters_so_far = args.iter_num
        timesteps_so_far = (
            args.iter_num * args.rollout_len * args.g_steps
        )  # missing g steps ?
        print(f"Resuming training from iteration {args.iter_num}")
        model_timeout_path = osp.join(args.model_path, "model_timeout")
        agent.load(model_timeout_path, args.iter_num, random_state=True, rank=rank)

    # Create collections
    d = defaultdict(list)
    eval_deque = deque(maxlen=args.maxlen)

    # Set up model save directory
    ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    if rank == 0:
        # Save the model as a dry run, to avoid bad surprises at the end
        agent.save(ckpt_dir, "{}_dryrun".format(iters_so_far))
        logger.info("dry run. Saving model @: {}".format(ckpt_dir))
        ckpt_dir_best = osp.join(ckpt_dir, "model_best")
        os.makedirs(ckpt_dir_best, exist_ok=True)
        if args.record:
            vid_dir = osp.join(args.video_dir, experiment_name)
            os.makedirs(vid_dir, exist_ok=True)

        # Handle timeout signal gracefully
        timeout_handler = TimeoutHandler(
            tstart=tstart,
            execution_time=args.execution_time,
            agent=agent,
            ckpt_dir=ckpt_dir,
            logger=logger,
        )

    # Create rollout generator for training the agent
    roll_gen = agent.rollout_generator()

    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        global RECORD
        RECORD = args.record
        eval_ep_gen = ep_generator(
            eval_env,
            agent,
            args.render,
        )

    # pretrain policy in bc manner
    if args.pretrain and not args.resume_training:
        agent.pretrain()

    while iters_so_far <= num_iters:
        log_iter_info(
            logger, iters_so_far, num_iters, tstart, timesteps=timesteps_so_far
        )

        # before was saving only for rank
        if iters_so_far % args.save_frequency == 0:
            # Save the model
            if rank == 0:
                agent.save(ckpt_dir, iters_so_far, random_state=True, rank=rank)
            else:
                agent.save_random_states(ckpt_dir, iters_so_far, rank=rank)
            logger.info("saving model @: {}".format(ckpt_dir))

        agent.do_iteration(
            timed=timed,
            roll_gen=roll_gen,
            iters_so_far=iters_so_far,
            d=d,
        )

        if eval_env is not None:
            assert rank == 0, "non-zero rank mpi worker forbidden here"

            if iters_so_far % args.eval_frequency == 0:
                with timed("evaluating"):
                    start_ = time.time()

                    eval_steps_per_iter = (
                        1 if benchmark == "dmc_ep" else args.eval_steps_per_iter
                    )

                    for _eval_step in range(eval_steps_per_iter):
                        # Sample an episode w/ non-perturbed actor w/o storing anything
                        if (
                            args.record
                            and iters_so_far % args.record_frequency == 0
                            and _eval_step < args.record_n_ep
                        ):
                            RECORD = True
                        else:
                            RECORD = False
                        agent.to("cpu")
                        eval_ep = eval_ep_gen.__next__()

                        # Aggregate data collected during the evaluation to the buffers
                        d["eval_len"].append(eval_ep["ep_lens"])
                        d["eval_env_ret"].append(eval_ep["ep_env_rets"])

                        if (
                            args.record
                            and iters_so_far % args.record_frequency == 0
                            and _eval_step < args.record_n_ep
                        ):
                            logger.info("recording episode @: {}".format(vid_dir))
                            # Record the last episode in a video
                            if benchmark == "dmc_ep":
                                for eval_step in range(args.record_n_ep):
                                    record_video(
                                        vid_dir,
                                        f"{iters_so_far}-{eval_step}",
                                        eval_ep["obs_render"][eval_step],
                                        record_format=args.record_format,
                                    )
                            else:
                                record_video(
                                    vid_dir,
                                    f"{iters_so_far}-{_eval_step}",
                                    eval_ep["obs_render"],
                                    record_format=args.record_format,
                                )

                    eval_env_ret = np.mean(d["eval_env_ret"])
                    eval_deque.append(eval_env_ret)

                    if agent.hps.algo == "bc":
                        import json

                        with open("data_bc.json", "r") as file:
                            data = json.load(file)

                        algo = agent.algo
                        data[algo][str(agent.hps.batch_size)][agent.hps.env_id][
                            str(agent.hps.seed)
                        ] = {
                            "return": {
                                "mean": np.mean(d["eval_env_ret"]),
                                "std": np.std(d["eval_env_ret"]),
                                "all": d["eval_env_ret"],
                            }
                        }
                        with open("data_bc.json", "w") as file:
                            json.dump(data, file, indent=4)

                    if eval_env_ret > best_eval_env_ret:
                        best_eval_env_ret = eval_env_ret
                        logger.info(f"saving best model @: {ckpt_dir_best}")
                        agent.save(ckpt_dir_best, "_best")
                        # save best model to wandb
                        for file in glob.glob(osp.join(ckpt_dir_best, "*_best.pth")):
                            os.makedirs(
                                os.path.join(wandb.run.dir, "model_best"), exist_ok=True
                            )
                            file = file.replace(ckpt_dir, "")[1:]
                            # shutil.copy(
                            #     osp.join(ckpt_dir, file),
                            #     os.path.join(wandb.run.dir, file),
                            # )
                        wandb.save(
                            osp.join(ckpt_dir_best, "*_best.pth"),
                            base_path=ckpt_dir,
                        )

        # Increment counters
        if hasattr(agent, "update_iters_so_far"):
            iters_so_far = agent.update_iters_so_far(iters_so_far)
        else:
            iters_so_far += 1
        timesteps_so_far = agent.update_timesteps(timesteps_so_far)

        if rank == 0 and (iters_so_far - 1) % args.eval_frequency == 0:
            # Log stats in csv
            logger.record_tabular("timestep", timesteps_so_far)
            logger.record_tabular("eval_len", np.mean(d["eval_len"]))
            logger.record_tabular("eval_env_ret", np.mean(d["eval_env_ret"]))
            logger.record_tabular("avg_eval_env_ret", np.mean(eval_deque))
            if agent.hps.kye_p and agent.hps.adaptive_aux_scaling:
                logger.record_tabular("cos_sim_p", np.mean(d["cos_sims_p"]))
            logger.info("dumping stats in .csv file")
            logger.dump_tabular()

            # Log stats in dashboard
            metrics = {"eval/avg_eval_env_ret": np.mean(eval_deque)}
            for k, v in d.items():
                if "eval" in k:
                    k = f"eval/{k}"
                metrics[k] = np.mean(v) if isinstance(v, list) else np.array(v)

            wandb.log(
                metrics,
                step=timesteps_so_far,
            )

            if args.algo in ["gail", "idmm", "sam-dac", "gail-bc"]:
                wandb.log({"dis_loss": np.mean(d["dis_losses"])}, step=timesteps_so_far)
                if agent.hps.kye_p and agent.hps.adaptive_aux_scaling:
                    wandb.log(
                        {"cos_sim_p": np.mean(d["cos_sims_p"])},
                        step=timesteps_so_far,
                    )

        # Clear the iteration's running stats
        d.clear()

        if rank == 0:
            # verify that there is still time remaining
            timeout_handler.step(iters_so_far)
        else:
            mpi_save_random_states(
                agent, osp.join(ckpt_dir, "model_timeout"), iters_so_far
            )

    if rank == 0:
        # verify that there is still time remaining
        timeout_handler.finish()
        # end the run
        logger.finish()
        # end wandb logging
        wandb.finish()
        logger.info("we're done. Saving model @: {}".format(ckpt_dir))
        logger.info("bye.")
    else:
        mpi_save_random_states(agent, osp.join(ckpt_dir, "model_timeout"), iters_so_far)
        wandb.finish()
