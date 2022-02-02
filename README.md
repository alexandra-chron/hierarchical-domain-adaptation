# Hierarchical Domain Adaptation with Adapters


Create a conda environment:

```
conda create -n hierarchical python=3.9.5
conda activate hierarchical
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

Install the rest of the requirements:

```
pip install -r ./requirements.txt
```


The DATA should be placed in ```hierarchical-domain-adaptation/corpora/```, in a ```name.{train,val,test}.json ``` format. 

The domains used are defined in ```domain_names.json```.
The names should match the names of the data in ```hierarchical-domain-adaptation/corpora/```.
Their number should match the ```--num_domains``` value.

The dictionary used is defined in ```domain_dict.json```. This is not needed for the baseline experiment. 

## 1. Get tree structure using GMMs and hierarchical clustering

Define the domains you will be using in ```domain_names_new.json```. Let's assume that you have 30 training domains. 

```

python3 run_clm_clusters.py --model_name_or_path gpt2 --output_dir ./dumped/fit_gmm_30_components  --do_eval --num_domains 30 --overwrite_output_dir --block_size 800 --per_device_eval_batch_size 1  --name fit_gmm_30_components_1000_seq_per_domain_pca_100 

```

This fits a GMM with *N* components (here, 30) to your data.
By default it uses 1000 sequences (of 800 tokens) per domain.
It creates a ```fit_gmm_30_components_1000_seq_per_domain_pca_100/``` directory, where you 
can find visualizations of the clustering (GMM + hierarchical) and the ```domain_to_cluster.json, domain_dict.json```,
which you should move to the main directory, as you will need them to train the hierarchical adapter model. 


## 2. Train hierarchical model 

```

python3 run_clm.py  --model_name_or_path gpt2 --do_train --do_eval --output_dir ./dumped/$YOUR_EXPERIMENT_NAME --num_domains 30 --num_train_epochs 20 --overwrite_output_dir --logging_strategy steps --logging_steps 900 --save_steps 900 --evaluation_strategy steps --logging_dir=logs/$YOUR_EXPERIMENT_NAME --block_size 800 --learning_rate 1e-3 --use_tree_structure True  --eval_steps 900 --load_best_model_at_end True --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --use_adapters True --adapter_size 64 --percentage_of_domain_in_cluster True > hierarchical_exp.log 

```

## 3. Evaluate hierarchical model in domain

```

python run_clm.py --model_name_or_path ./dumped/$YOUR_EXPERIMENT_NAME/ --do_eval --output_dir ./dumped/eval --use_adapters True --adapter_size 64 --use_tree_structure True --num_domains 30 --overwrite_output_dir --block_size 800 --per_device_eval_batch_size 16

```

## 4. (Optional)  Get domain-to-cluster allocation for out-of-domain evaluation
Assuming you also want to evaluate on held-out datasets (domains) (if not, skip this step), you need to define those domains in ```unseen_domains_new.json```.
Let's assume that you will evaluate on 10 held-out datasets (domains). 
 
```
python3 run_clm_clusters.py --model_name_or_path gpt2 --output_dir ./dumped/eval_ood --do_eval --num_domains 10 --overwrite_output_dir --block_size 800 --per_device_eval_batch_size 1 --find_clusters_for_unseen True --name eval10_doms --trained_gmm_path ./fit_gmm_30_components_1000_seq_per_domain_pca_100

```

It creates a ```eval10_doms/``` directory, where you can find the ```domain_to_cluster.json``` file, which you should copy to the main directory (```hierarchical-domain-adaptation/corpora/```) to run the out-of-domain evaluation of the trained model. 

## 5. (Optional)  Evaluate hierarchical model out of domain

Define the domains you want to evaluate on in the ```domain_names.json``` script (main directory).

```

python run_clm.py --model_name_or_path ./dumped/$YOUR_EXPERIMENT_NAME/ --do_eval --output_dir ./dumped/eval --use_adapters True --adapter_size 64 --use_tree_structure True --num_domains 10 --overwrite_output_dir --block_size 800 --per_device_eval_batch_size 16

```

## 6. (Optional) Run baseline (multi-domain adapters)

```

python3 run_clm.py  --model_name_or_path gpt2 --do_train --do_eval --output_dir ./dumped/$YOUR_EXPERIMENT_NAME --num_domains 1 --num_train_epochs 20 --overwrite_output_dir --logging_strategy steps --logging_steps 900 --save_steps 900 --evaluation_strategy steps --logging_dir=logs/$YOUR_EXPERIMENT_NAME --block_size 800 --learning_rate 1e-3 --use_tree_structure False  --eval_steps 900 --load_best_model_at_end True --per_device_train_batch_size 12 --per_device_eval_batch_size 6 --use_adapters True --adapter_size 512 --percentage_of_domain_in_cluster True > baseline_exp.log

```

## 7. (Optional) Evaluate GPT-2 (no adapters)

```

python3 run_clm.py --model_name_or_path gpt2 --do_eval --output_dir ./dumped/eval --use_adapters False --num_domains 30 --block_size 800   --use_tree_structure False

```
