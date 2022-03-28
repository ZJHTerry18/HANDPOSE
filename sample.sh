CUDA_VISIBLE_DEVICES=4 python train_gnn.py --num_process_per_node 1 --sample --port 10001 --save output324_gat_curl --latent_dims 8
# CUDA_VISIBLE_DEVICES=7 python train_2.py --num_process_per_node 1 --sample --port 10001 --save output317_pvae --latent_dims 5 --model PVAE
# CUDA_VISIBLE_DEVICES=4 python train.py --num_process_per_node 1 --sample --port 10001 --save output324_naive  --latent_dims 8