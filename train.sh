# CUDA_VISIBLE_DEVICES=5,6 python train_2.py --num_process_per_node 2 --save output317_pvae --port 10002 --latent_dims 5 --model PVAE
# CUDA_VISIBLE_DEVICES=4 python train.py --num_process_per_node 1 --save output324_naive --port 10002 --latent_dims 8 --batch_size 100 --learning_rate 1e-2
CUDA_VISIBLE_DEVICES=4 python train_gnn.py --num_process_per_node 1 --save output324_flat --port 10002 --latent_dims 8 --batch_size 1000 --learning_rate 1e-2