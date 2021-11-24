# Hierarchical Domain Adaptation (with Adapters) 


Create a conda environment:

```
conda create -n hierarchical python=3.9.5
conda activate hierarchical
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Install the rest of the requirements:

```
pip install -r ./requirements.txt
```


Before you run the following commands, make sure to set YOUR_EXPERIMENT_NAME, used to set the path below. This is the directory where results will be stored.

The DATA is already preprocessed and stored in ```/home/achron/hierarchical-domain-adaptation/corpora/cached_datasets``` (in vm-4 for example)

The domains used are defined in ```domain_names.json```. Their number should match the --num_domains value.

The dictionary used is defined in ```domain_dict.json```. This is not needed for the baseline experiment. 


## Run hierarchical model 

```

python3 run_clm.py  --model_name_or_path gpt2 --do_train --do_eval --output_dir ./dumped/$YOUR_EXPERIMENT_NAME --num_domains 39 --num_train_epochs 20 --overwrite_output_dir --logging_strategy steps --logging_steps 900 --save_steps 900 --evaluation_strategy steps --logging_dir=logs/$YOUR_EXPERIMENT_NAME --block_size 800 --learning_rate 1e-3 --use_tree_structure True  --eval_steps 900 --load_best_model_at_end True --per_device_train_batch_size 8 --per_device_eval_batch_size 6 --use_adapters True --adapter_size 64 --percentage_of_domain_in_cluster True > hierarchical_exp.log 

```


## Run baseline (multi-domain adapters)

```

python3 run_clm.py  --model_name_or_path gpt2 --do_train --do_eval --output_dir ./dumped/$YOUR_EXPERIMENT_NAME --num_domains 1 --num_train_epochs 20 --overwrite_output_dir --logging_strategy steps --logging_steps 900 --save_steps 900 --evaluation_strategy steps --logging_dir=logs/$YOUR_EXPERIMENT_NAME --block_size 800 --learning_rate 1e-3 --use_tree_structure False  --eval_steps 900 --load_best_model_at_end True --per_device_train_batch_size 8 --per_device_eval_batch_size 6 --use_adapters True --adapter_size 512 --percentage_of_domain_in_cluster True > baseline_exp.log

```

## Evaluate GPT-2 (no adapters)

```

python3 run_clm.py --model_name_or_path gpt2 --do_eval --output_dir ./dumped/eval --use_adapters False --num_domains 39 --block_size 800   --use_tree_structure False

```
