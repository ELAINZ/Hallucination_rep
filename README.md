This is the respository for reproducing CMU's paper Understanding Hallucinations in Diffusion Models through Mode Interpolation. 

fig10.py: the uncolored version
fig10_1.py: the colored version
other_examples.png: a sample of unclassified images

generate data: python gen_squares.py

train command: python main.py --arch UNet --dataset simple-shapes --data-dir simple-datasets/square_bottom_left_top_center_fixed --epochs 50 --batch-size 32 --diffusion-steps 1000 --sampling-steps 100 --num-sampled-images 500 --seed 1234 --exp_str test-single --log_results --filter_type "random" --start_timestep 700 --end_timestep 850 --num_timesteps 15

fig command: python fig10_1.py
