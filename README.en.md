# GridWord by DQN
Development of a GridWord environment to train a DQN (Deep Q-Network), which was also developed as part of the project.

## What is a DQN?
Deep Q-Learning is an extended version of the traditional Q-Learning algorithm, which uses deep neural networks to estimate Q-values.
Classic Q-Learning is effective in environments with a limited and fixed number of states, but it encounters problems with 
extensive or continuous state spaces due to the size of the Q-table. Deep Q-Learning overcomes this limitation by replacing the Q-table with a neural network 
capable of estimating Q-values for each state-action pair.

## Installation and Usage
To visualise DQN learning on the Gridword, you can run this project.

### Step 1: clone the project
```bash
  git clone 
```

### Step 2: create a Python environment
```bash
  python -m venv venv
  source ./venv/bin/activate
```

### Step 3: install dependencies
```bash
  pip install -r requirements.txt
```

### Step 4: run the project
```bash
  python train_dqn.py
```


Translated with DeepL.com (free version)
