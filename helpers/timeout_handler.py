import glob
import os
import os.path as osp
import shutil
import signal
import time

import wandb

from helpers.distributed_util import mpi_save_random_states
from helpers.misc_util import prettify_time
from helpers.mpi_util import MPI


def duration_to_seconds(duration):
    """Converts a duration in d-HH:MM:SS format to seconds."""
    if "-" in duration:
        duration = duration.split("-")
        days = duration[0]
        duration = duration[-1]
    else:
        days = 0
        duration = duration
    hours, mins, secs = duration.split(":")
    return int(days) * 86400 + int(hours) * 3600 + int(mins) * 60 + int(secs)


class TimeoutHandler:
    def __init__(
        self,
        tstart,
        execution_time,
        agent,
        ckpt_dir,
        logger,
        exit_duration=120,
    ):
        self.tstart = tstart
        self.execution_time = (
            duration_to_seconds(execution_time) if execution_time else None
        )
        self.agent = agent
        self.base_ckpt_dir = ckpt_dir
        self.ckpt_dir = osp.join(ckpt_dir, "model_timeout")
        self.logger = logger
        self.iters_so_far = 0
        self.exit_duration = exit_duration
        self._step_durations = []
        self._last_step_tstart = None
        self._status = "running"

        # self.setup_signals()

    def finish(self):
        # Save the model
        os.makedirs(self.ckpt_dir, exist_ok=True)
        mpi_save_random_states(
            self.agent,
            self.ckpt_dir,
            self.iters_so_far,
            save=True,
        )
        self.agent.save(self.ckpt_dir, self.iters_so_far)
        # save best model to wandb
        for file in glob.glob(osp.join(self.ckpt_dir, "*.pth")):
            file_name = file.split("/")[-1]
            timemout_folder = self.ckpt_dir.replace(f"{self.base_ckpt_dir}/", "")
            try:
                os.makedirs(osp.join(wandb.run.dir, timemout_folder), exist_ok=True)
                shutil.copy(
                    file, os.path.join(wandb.run.dir, timemout_folder, file_name)
                )
            except shutil.SameFileError:
                pass
        wandb.save(osp.join(self.ckpt_dir, "*.pth"), base_path=self.base_ckpt_dir)

    def timeout(self, signum, frame):
        self._status = "stopping"
        self.logger.info("Received kill signal, stopping execution...")
        self.logger.info(
            f"Time elapsed {0} in {1}".format(
                prettify_time(time.time() - self.tstart),
                prettify_time(self.execution_time),
            ),
        )
        self.logger.info(f"Signal information: {signum}, {frame}")

        self.finish()

        # end the run
        self.logger.finish()

        # end wandb logging
        wandb.finish()

        # exit mpi
        MPI.COMM_WORLD.Abort()

    def setup_signals(self):
        # Tie the timeout handler with the termination signal
        # Note, orterun relays SIGTERM and SIGINT to the workers as SIGTERM signals,
        # quickly followed by a SIGKILL signal (Open-MPI impl)
        signal.signal(signal.SIGTERM, self.timeout)
        signal.signal(signal.SIGUSR1, self.timeout)
        signal.signal(signal.SIGUSR2, self.timeout)

    def verify_execution_time(self):
        if self.execution_time is not None and self._status == "running":
            tallowed = self.execution_time - self.exit_duration
            # remove the maximum execution time for a step until now
            tallowed -= max(self._step_durations, default=0)
            if time.time() - self.tstart > tallowed:
                self.timeout("Manual Termination (execution time exceeded)", "=(")
            else:
                mpi_save_random_states(
                    self.agent,
                    self.ckpt_dir,
                    self.iters_so_far,
                    save=False,
                )

    def verify_discriminator_early_stopping(self):
        if self._status == "running":
            pass

    def verify_reward_early_stopping(self, reward):
        # args to be specified in the reward function
        tolerance = 0.1
        patience = 100
        self.expert_reward = 100
        if self._status == "running" and reward is not None:
            # if reward is close to expert reward, and no improvement has been made
            # for a given number of steps, stop training
            self._rewards.append(reward)
            self._rewards = self._rewards[-patience:]

            def is_reward_within_tolerance(current_reward, expert_reward, tolerance):
                """
                Check if the current reward is within a certain percentage (tolerance)
                of the expert's reward.

                Parameters:
                - current_reward (float): The current reward to be checked.
                - expert_reward (float): The expert's reward for comparison.
                - tolerance (float): The allowed percentage difference.

                Returns:
                - bool: True if the current reward is within the specified tolerance of the expert's reward, False otherwise.
                """
                return abs(current_reward / expert_reward) < 1 - tolerance

            # Check if the current reward is within a certain percentage (tolerance)
            # of the expert's reward.
            if is_reward_within_tolerance(reward, self.expert_reward, tolerance):
                if len(self._rewards) > patience and all(
                    is_reward_within_tolerance(reward, self.expert_reward, tolerance)
                    for reward in self._rewards
                ):
                    pass

    def step(self, step_size=None, reward=None, discriminator_loss=None):
        if self._last_step_tstart is None:
            self._last_step_tstart = time.time()
        else:
            self._step_durations.append(time.time() - self._last_step_tstart)

        if step_size is None:
            self.iters_so_far += 1
        else:
            self.iters_so_far = step_size

        self.verify_execution_time()
        # self.verify_discriminator_early_stopping(discriminator_loss)
        # self.verify_reward_early_stopping(reward)

        self._last_step_tstart = time.time()
        return False
