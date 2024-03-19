#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY=''

# TPU specific flags to improve training throughput
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

export PYTHONPATH="/home/ken/workspace/EasyLM/:$PYTHONPATH"

# on linux CPU remember to install CUDA
# Not sure if needed
# conda install nvidia/label/cuda-11.8.0::cuda-toolkit
# If there is cudnn mismatch
# pip uninstall nvidia-cudnn-cu116
# pip uninstall nvidia-cudnn-cu11
# pip install nvidia-cudnn-cu11==8.6.0.163
# pip uninstall fsspec
# pip uninstall gcsfs                                                         
# pip install -U datasets==2.16
# pip install gcsfs

# export LD_LIBRARY_PATH="/home/ken/anaconda3/envs/EasyLM/lib/python3.10/site-packages/nvidia/cudnn/lib/:/home/ken/anaconda3/envs/EasyLM/lib/:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="/home/ken/miniconda3/envs/EasyLM/lib/python3.10/site-packages/nvidia/cudnn/lib/:/home/ken/miniconda3/envs/EasyLM/lib/:/home/ken/miniconda3/envs/EasyLM/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="/home/ken/miniconda3/envs/EasyLM/lib/python3.10/site-packages/nvidia/cudnn/lib/:/home/ken/miniconda3/envs/EasyLM/lib/:/home/ken/miniconda3/envs/EasyLM/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ken/miniconda3/envs/EasyLM/lib/python3.10/site-packages/nvidia/cudnn/lib/:/home/ken/miniconda3/envs/EasyLM/lib/:$LD_LIBRARY_PATH"

# Traceback (most recent call last):
#   File "/home/ken/anaconda3/envs/EasyLM/lib/python3.10/runpy.py", line 196, in _run_module_as_main
#     return _run_code(code, main_globals, None,
#   File "/home/ken/anaconda3/envs/EasyLM/lib/python3.10/runpy.py", line 86, in _run_code
#     exec(code, run_globals)
#   File "/home/ken/workspace/EasyLM/EasyLM/models/llama/llama_train.py", line 267, in <module>
#     mlxu.run(main)
#   File "/home/ken/anaconda3/envs/EasyLM/lib/python3.10/site-packages/absl/app.py", line 308, in run
#     _run_main(main, args)
#   File "/home/ken/anaconda3/envs/EasyLM/lib/python3.10/site-packages/absl/app.py", line 254, in _run_main
#     sys.exit(main(argv))
#   File "/home/ken/workspace/EasyLM/EasyLM/models/llama/llama_train.py", line 209, in main
#     mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
#   File "/home/ken/workspace/EasyLM/EasyLM/models/llama/llama_model.py", line 242, in get_jax_mesh
#     return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'mp'))
#   File "/home/ken/workspace/EasyLM/EasyLM/jax_utils.py", line 168, in get_jax_mesh
#     mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
# ValueError: cannot reshape array of size 1 into shape (64,1)

python -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,1,1' \
    --dtype='fp32' \
    --total_steps=250000 \
    --log_freq=50 \
    --save_model_freq=0 \
    --save_milestone_freq=2500 \
    --load_llama_config='tiny64M' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.vocab_file='../llama-tokenizer/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='raw_content' \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.batch_size=8 \
    --train_dataset.huggingface_dataset.path='togethercomputer/RedPajama-Data-V2' \
    --train_dataset.huggingface_dataset.name='sample' \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_7b" \
    --logger.output_dir="../checkpoint" \
    --logger.wandb_dir="$HOME/experiment_output/open_llama_7b" \
|& tee output.txt
# |& tee $HOME/output.txt

