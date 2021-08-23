CUDA_VISIBLE_DEVICES=2 python main_mbpo.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch'

CUDA_VISIBLE_DEVICES=3 python main_mbpo_modular.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --pred_hidden_size 64

CUDA_VISIBLE_DEVICES=2 python main_mbpo_modular_rnn.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --pred_hidden_size 64

CUDA_VISIBLE_DEVICES=6 python main_mbpo_modular_role.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --pred_hidden_size 64 --num_networks 5 --num_elites 3

CUDA_VISIBLE_DEVICES=6 python main_mbpo_modular_role.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --pred_hidden_size 64 --num_networks 3 --num_elites 2

CUDA_VISIBLE_DEVICES=5 python main_mbpo_modular_role.py --env_name 'Lift' --num_epoch 300 --model_type 'pytorch' --num_networks 5 --num_elites 3 --pred_hidden_size 64