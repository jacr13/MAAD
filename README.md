# MiloZero

## Linting and formatting
Please lint your code before pushing it to the main branch.
The super_linter.sh script will do a scan over the repository and tell you the files
that should be reworked.
```bash
pip install flake8 flake8_commas flake8-black black
# or 
./super_linter.sh
```

## Running Experiments

<details style="border:1px solid #aaa; border-radius: 4px; padding: .5em .5em 0;">
  <summary style="font-size:20px; font-weight:bold; margin:-.5em -.5em 0; padding:.5em;">
    Trainning a single experiment
  </summary>

    * Direct
    ```bash
    python

    # or with docker
    docker run -e OMPI_MCA_plm_rsh_agent=sh \
      -e WANDB_API_KEY=wandb_key \
      -e WANDB_BASE_URL=https://api.wandb.ai \
      -v $(pwd):/workspace \
      candidj0/milozero:latest mpiexec -n 3 python main.py \
      --wandb_project=test-name-project \
      --wandb_entity=wandb_username \
      --task=train \
      --algo=ppo \
      --execution_time=4-00:00:00 \
      --save_frequency=50 \
      --p_perception_stack=mlp_100 \
      --v_perception_stack=mlp_100 \
      --d_perception_stack=mlp_100 \
      --batch_norm_all \
      --p_lr=0.0001 \
      --v_lr=0.0001 \
      --clip_norm=0.5 \
      --rollout_len=2048 \
      --batch_size=64 \
      --num_demos=16 \
      --uuid=some_name.seed02 \ # remove to generate it automatically
      --env_id=Hopper \
      --expert_path=data/demos/Hopper-v3 \
      --seed=2
    ```
</details>


<details open style="border:1px solid #aaa; border-radius: 4px; padding: .5em .5em 0;">
  <summary style="font-size:20px; font-weight:bold; margin:-.5em -.5em 0; padding:.5em;">
    Using the spawner (for local and cluster experiments)
  </summary>

  While one can launch any job via `main.py`, it is advised to use `spawner.py`,
  designed to spawn a swarm of experiments over multiple seeds and environments in one command.
  To get its usage description, type `python spawner.py -h`.
  ```bash
usage: spawner.py [-h] [--prefix PREFIX] [--config CONFIG]
                  [--conda_env CONDA_ENV] [--env_bundle ENV_BUNDLE]
                  [--args_method {full,yaml,minimal}] [--demo_dir DEMO_DIR]
                  [--num_workers NUM_WORKERS] [--num_trials NUM_TRIALS]
                  [--deployment {tmux,slurm}] [--num_seeds NUM_SEEDS]
                  [--caliber {debug,veryveryshort,veryshort,short,long,daylong,verylong,veryverylong,veryveryverylong}]
                  [--deploy_now] [--no-deploy_now] [--sweep] [--no-sweep]
                  [--clear] [--no-clear] [--wandb_upgrade]
                  [--no-wandb_upgrade] [--num_demos NUM_DEMOS [NUM_DEMOS ...]]
                  [--wandb_base_url WANDB_BASE_URL]
                  [--wandb_api_key WANDB_API_KEY]
                  [--wandb_entity WANDB_ENTITY]
                  [--wandb_project WANDB_PROJECT] [--docker] [--no-docker]
                  [--docker_image DOCKER_IMAGE]

Job Spawner

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX
  --config CONFIG
  --conda_env CONDA_ENV
  --env_bundle ENV_BUNDLE
  --args_method {full,yaml,minimal}
                        Choose which parameters to use in command
  --demo_dir DEMO_DIR
  --num_workers NUM_WORKERS
  --num_trials NUM_TRIALS
  --deployment {tmux,slurm}
                        deploy how?
  --num_seeds NUM_SEEDS
  --caliber {debug,veryveryshort,veryshort,short,long,daylong,verylong,veryverylong,veryveryverylong}
  --deploy_now          deploy immediately?
  --no-deploy_now
  --sweep               hp search?
  --no-sweep
  --clear               clear files after deployment
  --no-clear
  --wandb_upgrade       upgrade wandb?
  --no-wandb_upgrade
  --num_demos NUM_DEMOS [NUM_DEMOS ...], --list NUM_DEMOS [NUM_DEMOS ...]
  --wandb_base_url WANDB_BASE_URL
                        your wandb base url
  --wandb_api_key WANDB_API_KEY
                        your wandb api key
  --wandb_entity WANDB_ENTITY
                        team or personal username
  --wandb_project WANDB_PROJECT
                        wandb project name
  --docker              use docker?
  --no-docker
  --docker_image DOCKER_IMAGE
                        Name of docker image or path to image (cluster)
  ```

  Here is an example:
  ```bash
  python spawner.py \  #
    --config tasks/train_mujoco_ppo.yaml \  # config file describing the hps
    --env_bundle debug \  # environement bundle for experiments
    --wandb_upgrade \  # should we upgrade wandb?
    --no-sweep \  # do not do hps search
    --clear \  # after deployment clean launching files
    --deploy_now \  # run the experiments now ? (if not files to launch are saved)
    --caliber short \  # name corresponding to a duration
    --num_workers 2 \  # number of workers to use
    --num_seeds 3 \  # number of seeds for the same experiment
    --deployment tmux \  # choose if you want to deploy local (tmux) or in the cluster (slurm)
    --conda_env pytorch \  # name of conda environment (not needed for docker or if current env is the same)
    --wandb_base_url [YOUR_WANDB_BASE_URL] \  # wandb url (only for personal instances)
    --wandb_api_key [YOUR_WANDB_API_KEY] \  # wandb key
    --wandb_entity [YOUR_WANDB_ENTITY] \  # wandb username or team name
    --docker \  # launch with docker
    --docker_image docker_user/milozero:mujoco  # the docker image name
  ```

  With docker you need to add this arguments (as in the example above):
  ```bash
  --docker --docker_image=docker_user/milozero:mujoco
  ```

  When using sweep, you need to specify the the searching arguments between `[]` in the config file,
  everytime the spawner will see a list it will randomly choose one element form it.
  In addition you need to specify the maximum number of trials (maximum number of combinations) for your hyperparameter search.

  Please note that the maximum number of experiments will be:
  experiment_total = num_trial * num_seeds

  For an example of a config file using sweep, please see [this file.](./tasks/train_mujoco_ppo_hps.yaml)
</details>

<details open style="border:1px solid #aaa; border-radius: 4px; padding: .5em .5em 0;">
  <summary style="font-size:20px; font-weight:bold; margin:-.5em -.5em 0; padding:.5em;">
    Evaluation
  </summary>

  ### From wandb (recommended)
  ```bash
  WANDB_RUN_PATH="username/wandb_project/experiment_name"
  python main.py \
   --task=eval \  # do only evaluation
   --wandb_project=SOME_NAME \  # some name for the evaluation
   --wandb_run_path=$WANDB_RUN_PATH \  # path to the experiment in wandb
   --wandb_download_replace \  # if model already exists, replace it
   --iter_num='best' \  # use best model
   --num_trajs=16  # number of trajectories to launch
  ```
  ### From local files
  ```bash
  python main.py \
   --task=eval \  # do only evaluation
   --wandb_project=SOME_NAME \  # some name for the evaluation
   --model_path=path_to_model/checkpoints/experiment_name \
   --iter_num=620 \  # iteration of the model
   --num_trajs=1 \  # number of trajectories to launch
   --render \  # render the trajectories (may require export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so)
   --save_traj  # save the trajectory
  ```
</details>


<details style="border:1px solid #aaa; border-radius: 4px; padding: .5em .5em 0;">
  <summary style="font-size:20px; font-weight:bold; margin:-.5em -.5em 0; padding:.5em;">
    Dependencies (Should be updated)
  </summary>

  ### Without conda
  ```bash
  # create a virtual environment
  python -m venv venv
  # activate it
  source venv/bin/activate
  # install requirements
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

  If you get an error installing mpi4py, you are probably missing libopenmpi-dev. Try:
  ```bash
  sudo apt -y install libopenmpi-dev
  ```
  ### With conda
  ```bash
  # create a virtual environment and install requirements
  conda create -y -n venv python=3.7
  # activate it
  conda activate venv
  # install requirements
  while read requirement; do if [[ $requirement != "#"* ]]; then conda install -c conda-forge --yes $requirement || pip install $requirement; fi; done < requirements.txt
  ```
  ### Baobab
  ```bash
  ./setup_env_baobab.sh
  ```
</details>

<details style="border:1px solid #aaa; border-radius: 4px; padding: .5em .5em 0;">
  <summary style="font-size:20px; font-weight:bold; margin:-.5em -.5em 0; padding:.5em;">
    Flow of execution
  </summary>

  This graphic describes the basic flow of execution for training an agent.

  ![Flow chart describing the execution sequence of files and functions](assets/images/flow_chart.png)

  `main.py`  will  begin  to  initialize  an  experiment  from  the  class
  `ExperimentInitializer`. Inside will be stocked the parameters of the experiment
  and  his  name.  After  that,  `main.py`  will  instantiate  the  environment  for  the
  experiment  and  it  can  be  a  mujoco,  opensim  or  dmc  environment.  Then  the
  agent will be defined and for that, the environment is needed.

  Now, the training of the agent can start and `orchestrator.py` will orchestrate all
  the work by calling the right functions from the agent, in this example `ppo_agent.py`.
  Starting by initialized `rollout_generator()` who will be used later for sampling states actions pairs.
  Then `orchestrator.py` will start to train the agent trough the function `do_iteration()`.

  This  method  start  to  sample  some  state  action  pairs  thanks  to  `rollout_generator()`
  and then update the agent with the data just sampled. `do_iteration()`
  will be repeated until the number of iterations is equal to the variable `num_iters`.
</details>
