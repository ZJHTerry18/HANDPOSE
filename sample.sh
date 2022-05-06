# CUDA_VISIBLE_DEVICES=4 python train_gnn.py --num_process_per_node 1 --sample --port 10000 --save gat_new_model --latent_dims 8 
# CUDA_VISIBLE_DEVICES=7 python train_2.py --num_process_per_node 1 --sample --port 10001 --save output317_pvae --latent_dims 5 --model PVAE
# CUDA_VISIBLE_DEVICES=4 python train.py --num_process_per_node 1 --sample --port 10001 --save output324_naive  --latent_dims 8 --data_vis
# CUDA_VISIBLE_DEVICES=5 python train_finger.py --num_process_per_node 1 --sample --port 10001 --cont_training --save output_fingernet_on_feat
# CUDA_VISIBLE_DEVICES=5 python train_hi_mlp.py --num_process_per_node 1 --sample --port 10001 --save output_hi_vae_normal --latent_dims 8 --batch_size 100 --learning_rate 1e-3 --epochs 1000
CUDA_VISIBLE_DEVICES=5 python train_nvae_mlp.py --num_process_per_node 1 --sample --port 10001 --save output_nvae_vae --latent_dims 4 --batch_size 100 --learning_rate 1e-3 --epochs 1000
# CUDA_VISIBLE_DEVICES=5 python train_finger2.py  --num_process_per_node 1 --sample --port 10001 --save output_fingernet_wo_vae --latent_dims 4 --batch_size 100 --learning_rate 1e-3 --epochs 1000