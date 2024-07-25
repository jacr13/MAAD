# MAAD-AIL
mpiexec -n 4 python main.py \
    --env_id=Walker-Walk-Feat-v0 \
    --seed=1 \
    --expert_path=data/demos/dmc/Walker-Walk-Feat-v0 \
    --wandb_project=MAAD_DMC \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=maad \
    --execution_time=4-00:00:00 \
    --p_perception_stack=mlp_256 \
    --v_perception_stack=mlp_128 \
    --d_perception_stack=mlp_128 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --eps=0.1 \
    --idm_mdn_n_gaussians=1 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl


# GAIL-BC
mpiexec -n 4 python main.py \
    --env_id=Walker-Walk-Feat-v0 \
    --seed=1 \
    --expert_path=data/demos/dmc/Walker-Walk-Feat-v0 \
    --wandb_project=MAAD_DMC \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=gail-bc \
    --execution_time=4-00:00:00 \
    --p_perception_stack=mlp_128 \
    --v_perception_stack=mlp_128 \
    --d_perception_stack=mlp_128 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --eps=0.1 \
    --idm_mdn_n_gaussians=1 \
    --d_lr=0.0001 \
    --num_demos=16 \
    --d_batch_norm \
    --gail_bc_lambda_decay=1


# GAIfO
mpiexec -n 4 python main.py \
    --env_id=Walker-Walk-Feat-v0 \
    --seed=1 \
    --expert_path=data/demos/dmc/Walker-Walk-Feat-v0 \
    --wandb_project=MAAD_DMC \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=gail \
    --execution_time=4-00:00:00 \
    --p_perception_stack=mlp_256 \
    --v_perception_stack=mlp_128 \
    --d_perception_stack=mlp_128 \
    --batch_norm_all \
    --p_batch_norm \
    --v_batch_norm \
    --p_lr=0.0001 \
    --v_lr=0.0001 \
    --clip_norm=0.5 \
    --lr_schedule=linear \
    --g_steps=9 \
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --bc_epochs=1000


# MAAD-OT
mpiexec -n 4 python main.py \
    --env_id=Walker-Walk-Feat-v0 \
    --seed=1 \
    --expert_path=data/demos/dmc/Walker-Walk-Feat-v0 \
    --wandb_project=MAAD_DMC \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=maad-ot \
    --execution_time=4-00:00:00 \
    --p_perception_stack=mlp_128 \
    --v_perception_stack=mlp_128 \
    --d_perception_stack=mlp_128 \
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
    --bc_epochs=1000 \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl \
    --ot_rewards=sinkhorn_cosine


# MAAD-TM
mpiexec -n 4 python main.py \
    --env_id=Walker-Walk-Feat-v0 \
    --seed=1 \
    --expert_path=data/demos/dmc/Walker-Walk-Feat-v0 \
    --wandb_project=MAAD_DMC \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=maad-ot \
    --execution_time=4-00:00:00 \
    --p_perception_stack=mlp_128 \
    --v_perception_stack=mlp_128 \
    --d_perception_stack=mlp_128 \
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
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --bc_epochs=1000 \
    --gail_bc_lambda_decay=1


# TMO
mpiexec -n 4 python main.py \
    --env_id=Walker-Walk-Feat-v0 \
    --seed=1 \
    --expert_path=data/demos/dmc/Walker-Walk-Feat-v0 \
    --wandb_project=MAAD_DMC \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=tmot \
    --execution_time=4-00:00:00 \
    --p_perception_stack=mlp_128 \
    --v_perception_stack=mlp_128 \
    --d_perception_stack=mlp_128 \
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
    --bc_epochs=1000 \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl


# OTO
mpiexec -n 4 python main.py \
    --env_id=Walker-Walk-Feat-v0 \
    --seed=1 \
    --expert_path=data/demos/dmc/Walker-Walk-Feat-v0 \
    --wandb_project=MAAD_DMC \
    --wandb_entity=<your_wandb_entity> \
    --task=train \
    --algo=tmot \
    --execution_time=4-00:00:00 \
    --p_perception_stack=mlp_256 \
    --v_perception_stack=mlp_128 \
    --d_perception_stack=mlp_128 \
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
    --d_lr=0.0001 \
    --state_state \
    --num_demos=16 \
    --d_batch_norm \
    --bc_epochs=1000 \
    --gail_bc_lambda_decay=1 \
    --gail_bc_loss=forward_kl \
    --ot_rewards=sinkhorn_cosine


