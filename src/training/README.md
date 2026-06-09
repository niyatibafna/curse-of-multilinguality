## Training experiments with adding more languages

The idea is to train many multilingual models with the same token budget / additive token budget, but of varying number of languages (1, 5, 10, 25, 50, 75, 100). 

We will then compute our metrics on these models, characterising the curse of multilinguality for embedding spaces.

### Data
We will get data from MADLAD400. 

### Languages
We choose 100 languages from the dataset, not including English. We'll then start with a random subset, and then add more languages to it to create incrementally bigger language subsets.

### Data sampling strategies
#### Fixed token budget
We'll have a fixed token budget of 20M, and split it equally across all languages in the group

#### Additive
We will sample 1M tokens per language and fix this.
The dataset size increases as we add more languages.

### Model and training
We'll do a 4-layer BERT-style Transformer with size hidden size 256 and 4 heads.
We'll train a tokenizer with vocabulary 50k on 100M tokens across 100 languages.
We always use the same tokenizer regardless of language group or data sampling strategy.

For each language subset, we train this model from scratch for 3 epochs.

### Optimization and other hyperparameters
We use default learning rate and other hyperparameters. 

### Checkpoints and storage
Use an appropriate place in $DATADIR as per conventions for data and models.


## Todos
### Model stream
Prepare training scripts and infra.

### Data stream
Download data, prepare sampling and loading scripts. If possible, only download as much data as needed.

### Tokenizer stream
Train tokenizer (use BERT-defaults)

### Training stream
Prepare jobs, monitor, debug

### Evaluation stream
Prepare model loading scripts and extraction of embeddings for sentences, ready for metric computation in line with current repo.

### Documentation
Use this README for instructions to agents, and status recording.