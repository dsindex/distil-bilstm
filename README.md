# how to distil bert to lstm

```
# prerequisites

$ python -m pip install -r requirements
$ python -m spacy download en_core_web_sm


# train bert

$ python train_bert.py --data_dir sst2 --output_dir bert_output --epochs 3 --batch_size 64 --lr 1e-5 --lr_schedule warmup --warmup_steps 100 --do_train
{'loss': 0.22899218350135636, 'perplexity': 1.2573322110359069, 'accuracy': 0.9299655568312285}

# generating pseudo labeled 'data + augmented data'

$ python generate_dataset.py --input sst2/train.tsv --output sst2/augmented.tsv --model bert_output
$ wc -l sst2/train.tsv sst2/augmented.tsv
   67349 sst2/train.tsv
 1019579 sst2/augmented.tsv
$ more sst2/augmented.tsv
...
remains utterly satisfied to remain the same throughout	0.895635 -0.408034
remains utterly satisfied to remain <mask> same throughout	1.753451 -1.064335
remains <mask> repulsive to remain the same throughout	2.333677 -2.028693
remains <mask> satisfied within remain both same throughout	-0.424739 0.417799
remains utterly satisfied <mask> <mask> <mask> <mask> throughout	-0.516992 0.773238
remains utterly satisfied to <mask> the same anew	0.589114 -0.107572
remains utterly satisfied to <mask> the same <mask>	1.578492 -0.938877
<mask> utterly satisfied to remain the same throughout	1.845652 -1.405137
remains utterly fourth to remain the same <mask>	2.092397 -1.506175
<mask> utterly satisfied to remain the same <mask>	1.235599 -0.915951
remains utterly satisfied <mask> remain the same throughout	0.679312 -0.015502
remains utterly satisfied to remain the <mask> <mask>	1.097267 -0.658418
inc utterly satisfied to remain these same throughout	-0.028254 0.373944
<mask> utterly satisfied to remain both same <mask>	1.359792 -0.923046
remains <mask> satisfied <mask> <mask> the same throughout	1.752900 -1.216693
...


# train bilstm with train.tsv

$ python train_bilstm.py --data_dir sst2 --output_dir bilstm_output --epochs 3 --batch_size 50 --lr 1e-3 --lr_schedule warmup --warmup_steps 100 --do_train
$ {'loss': 0.5686685516147197, 'perplexity': 1.7659142617799288, 'accuracy': 0.8231917336394948}


# train bilstm with augmented.tsv

$ python train_bilstm.py --data_dir sst2 --output_dir bilstm_output --epochs 3 --batch_size 50 --lr 1e-3 --lr_schedule warmup --warmup_steps 100 --do_train --augmented
{'loss': 0.34691110516798895, 'perplexity': 1.4146909610652816, 'accuracy': 0.886337543053961}

```


----

# distil-bilstm

[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run)

This repository contains scripts to train a tiny bidirectional LSTM classifier on the SST-2 dataset (url).
It also contains a script to fine-tune `bert-large-uncased` on the same task.
The procedure is inspired by the paper [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136).

### Installing requirements

```bash
pip install -r requirements.txt  # Skip this if you are running on FloydHub
python -m spacy download en
```

### Fine-tuning bert-large-uncased


```bash
>> python train_bert.py --help

usage: train_bert.py [-h] --data_dir DATA_DIR --output_dir OUTPUT_DIR
                     [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                     [--lr_schedule {constant,warmup,cyclic}]
                     [--warmup_steps WARMUP_STEPS]
                     [--epochs_per_cycle EPOCHS_PER_CYCLE] [--do_train]
                     [--seed SEED] [--no_cuda] [--cache_dir CACHE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing the dataset.
  --output_dir OUTPUT_DIR
                        Directory where to save the model.
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR               Learning rate.
  --lr_schedule {constant,warmup,cyclic}
                        Schedule to use for the learning rate. Choices are:
                        constant, linear warmup & decay, cyclic.
  --warmup_steps WARMUP_STEPS
                        Warmup steps for the 'warmup' learning rate schedule.
                        Ignored otherwise.
  --epochs_per_cycle EPOCHS_PER_CYCLE
                        Epochs per cycle for the 'cyclic' learning rate
                        schedule. Ignored otherwise.
  --do_train
  --seed SEED           Random seed.
  --no_cuda
  --cache_dir CACHE_DIR
                        Custom cache for transformer models.
```

Example:

```bash
python train_bert.py --data_dir SST-2 --output_dir bert_output --epochs 1 --batch_size 16 --lr 1e-5 --lr_schedule warmup --warmup_steps 100 --do_train
```

### Generating the augmented dataset

The file used in my tests is available at https://www.floydhub.com/alexamadori/datasets/sst-2-augmented/1, but you may want to generate another one with a random seed or to use a different teacher model.

```bash
>> python generate_dataset.py --help

usage: generate_dataset.py [-h] --input INPUT --output OUTPUT --model MODEL
                           [--no_augment] [--batch_size BATCH_SIZE]
                           [--no_cuda]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input dataset.
  --output OUTPUT       Output dataset.
  --model MODEL         Model to use to generate the labels for the augmented
                        dataset.
  --no_augment          Don't perform data augmentation
  --batch_size BATCH_SIZE
  --no_cuda

```

Example:

```bash
python generate_dataset.py --input SST-2/train.tsv --output SST-2/augmented.tsv --model bert_output
```

### Training the BiLSTM model

```bash
>> python train_bilstm.py --help

usage: train_bilstm.py [-h] --data_dir DATA_DIR --output_dir OUTPUT_DIR
                       [--augmented] [--epochs EPOCHS]
                       [--batch_size BATCH_SIZE] [--lr LR]
                       [--lr_schedule {constant,warmup,cyclic}]
                       [--warmup_steps WARMUP_STEPS]
                       [--epochs_per_cycle EPOCHS_PER_CYCLE] [--do_train]
                       [--seed SEED] [--no_cuda]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing the dataset.
  --output_dir OUTPUT_DIR
                        Directory where to save the model.
  --augmented           Wether to use the augmented dataset for knowledge
                        distillation
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR               Learning rate.
  --lr_schedule {constant,warmup,cyclic}
                        Schedule to use for the learning rate. Choices are:
                        constant, linear warmup & decay, cyclic.
  --warmup_steps WARMUP_STEPS
                        Warmup steps for the 'warmup' learning rate schedule.
                        Ignored otherwise.
  --epochs_per_cycle EPOCHS_PER_CYCLE
                        Epochs per cycle for the 'cyclic' learning rate
                        schedule. Ignored otherwise.
  --do_train
  --seed SEED
  --no_cuda
```

Example:

```bash
python train_bilstm.py --data_dir SST-2 --output_dir bilstm_output --epochs 1 --batch_size 50 --lr 1e-3 --lr_schedule warmup --warmup_steps 100 --do_train --augmented
```
