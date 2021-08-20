CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch'

CUDA_VISIBLE_DEVICES=7 python main_mbpo_modular.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --pred_hidden_size 64

CUDA_VISIBLE_DEVICES=0 python main_mbpo_modular_role.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --pred_hidden_size 64 --num_networks 5 --num_elites 3

CUDA_VISIBLE_DEVICES=5 python main_mbpo_modular_role.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --num_networks 5 --num_elites 3 --pred_hidden_size 64