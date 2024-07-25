# GAIfO
mpiexec -n 4 python main.py \
    --env_id=Ant-v3 \
    --seed=1 \
    --expert_path=data/demos/Ant-v3 \
    --wandb_project=MAAD_GYM \
    --wandb_entity=<your_wandb_entity> \
    --checkpoint_dir=data/checkpoints \
    --log_dir=data/logs \
    --video_dir=data/videos \
    --task=train \
    --algo=gail \
    --execution_time=4-00:00:00 \
    --training_steps_per_iter=4 \
    --p_perception_stack=mlp_100 \
    --v_perception_stack=mlp_100 \
    --d_perception_stack=mlp_100 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --eps=0.1 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm


# MAAD-AIL
mpiexec -n 4 python main.py \
    --env_id=Ant-v3 \
    --seed=1 \
    --expert_path=data/demos/Ant-v3 \
    --wandb_project=MAAD_GYM \
    --wandb_entity=<your_wandb_entity> \
    --checkpoint_dir=data/checkpoints \
    --log_dir=data/logs \
    --video_dir=data/videos \
    --task=train \
    --algo=gail-bco \
    --execution_time=2-00:00:00 \
    --training_steps_per_iter=4 \
    --p_perception_stack=mlp_100 \
    --v_perception_stack=mlp_100 \
    --d_perception_stack=mlp_100 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --eps=0.1 \
    --idm_mdn_n_gaussians=2 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda=10 \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl


# MAAD-OT
mpiexec -n 4 python main.py \
    --env_id=Ant-v3 \
    --seed=1 \
    --expert_path=data/demos/Ant-v3 \
    --wandb_project=MAAD_GYM \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=maad-ot \
    --execution_time=4-00:00:00 \
    --num_timesteps=21000000.0 \
    --p_perception_stack=mlp_100 \
    --v_perception_stack=mlp_100 \
    --d_perception_stack=mlp_100 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --rollout_len=1000 \
    --eps=0.1 \
    --idm_mdn_n_gaussians=1 \
    --g_steps=6 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda=10 \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl \
    --ot_rewards=sinkhorn_cosine \
    --no-rot_auto_rew_scale


# MAAD-TM
mpiexec -n 4 python main.py \
    --env_id=Ant-v3 \
    --seed=1 \
    --expert_path=data/demos/Ant-v3 \
    --wandb_project=MAAD_GYM \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=maad-ot \
    --execution_time=4-00:00:00 \
    --num_timesteps=21000000.0 \
    --p_perception_stack=mlp_100 \
    --v_perception_stack=mlp_100 \
    --d_perception_stack=mlp_100 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --rollout_len=1000 \
    --eps=0.1 \
    --idm_mdn_n_gaussians=1 \
    --g_steps=6 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda=10 \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl \
    --no-rot_auto_rew_scale


# GAIL-BC
mpiexec -n 4 python main.py \
    --env_id=Ant-v3 \
    --seed=1 \
    --expert_path=data/demos/Ant-v3 \
    --wandb_project=MAAD_GYM \
    --wandb_entity=<your_wandb_entity> \
    --checkpoint_dir=data/checkpoints \
    --log_dir=data/logs \
    --video_dir=data/videos \
    --task=train \
    --algo=gail-bc \
    --execution_time=4-00:00:00 \
    --training_steps_per_iter=4 \
    --p_perception_stack=mlp_100 \
    --v_perception_stack=mlp_100 \
    --d_perception_stack=mlp_100 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --eps=0.1 \
    --d_lr=0.0001 \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda=10


# TMO
mpiexec -n 4 python main.py \
    --env_id=Ant-v3 \
    --seed=1 \
    --expert_path=data/demos/Ant-v3 \
    --wandb_project=MAAD_GYM \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=tmot \
    --execution_time=0-16:00:00 \
    --num_timesteps=21000000.0 \
    --p_perception_stack=mlp_100 \
    --v_perception_stack=mlp_100 \
    --d_perception_stack=mlp_100 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --rollout_len=1000 \
    --idm_mdn_n_gaussians=1 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda=10 \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl \
    --no-rot_auto_rew_scale


# OTO
mpiexec -n 4 python main.py \
    --env_id=Ant-v3 \
    --seed=1 \
    --expert_path=data/demos/Ant-v3 \
    --wandb_project=MAAD_GYM \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=tmot \
    --execution_time=4-00:00:00 \
    --num_timesteps=21000000.0 \
    --p_perception_stack=mlp_100 \
    --v_perception_stack=mlp_100 \
    --d_perception_stack=mlp_100 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --rollout_len=1000 \
    --idm_mdn_n_gaussians=1 \
    --g_steps=6 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl \
    --ot_rewards=sinkhorn_cosine \
    --no-rot_auto_rew_scale


