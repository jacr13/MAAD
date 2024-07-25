# Mimicking Better by Matching the Approximate Action Distribution

Code for the paper:  [Mimicking Better by Matching the Approximate Action Distribution](https://proceedings.mlr.press/v235/candido-ramos24a.html)

## Running Experiments

<details style="border:1px solid #aaa; border-radius: 4px; padding: .5em .5em 0;">
  <summary style="font-size:20px; font-weight:bold; margin:-.5em -.5em 0; padding:.5em;">
    Trainning a single experiment
  </summary>

    * Direct
    ```bash
    # Check tasks folder to see specific commands
    mpiexec -n 4 python main.py \
      ...

    # or with docker
    docker run -e OMPI_MCA_plm_rsh_agent=sh \
      -e WANDB_API_KEY=wandb_key \
      -e WANDB_BASE_URL=https://api.wandb.ai \
      -v $(pwd):/workspace \
      candidj0/milozero:latest mpiexec -n 4 python main.py \
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
    Evaluation
  </summary>

  ### From wandb (recommended)
  ```bash
  WANDB_RUN_PATH="username/wandb_project/experiment_name"
  python main.py \
   --task=eval \  # do only evaluation
   --wandb_project=<SOME_NAME> \  # some name for the evaluation
   --wandb_run_path=$WANDB_RUN_PATH \  # path to the experiment in wandb
   --wandb_download_replace \  # if model already exists, replace it
   --iter_num='best' \  # use best model
   --num_trajs=16  # number of trajectories to launch
  ```
  ### From local files
  ```bash
  python main.py \
   --task=eval \  # do only evaluation
   --wandb_project=<SOME_NAME> \  # some name for the evaluation
   --model_path=path_to_model/checkpoints/experiment_name \
   --iter_num=620 \  # iteration of the model or 'best' for best model
   --num_trajs=1 \  # number of trajectories to launch
   --render \  # render the trajectories (may require export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so)
   --save_traj  # save the trajectory
  ```
</details>


<details style="border:1px solid #aaa; border-radius: 4px; padding: .5em .5em 0;">
  <summary style="font-size:20px; font-weight:bold; margin:-.5em -.5em 0; padding:.5em;">
    Dependencies
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
</details>

