CUDA_VISIBLE_DEVICES=7 python main_mbpo.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch'

CUDA_VISIBLE_DEVICES=5 python main_mbpo_modular.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch'

CUDA_VISIBLE_DEVICES=2 python main_mbpo_modular_role.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch'