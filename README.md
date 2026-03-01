# GridWord par DQN
Réalisation d'un environnement Gridword afin d'entrainer un DQN (Deep Q-Network) lui aussi réalisé dans le projet.

## Qu'est ce qu'un DQN ?
Le Deep Q-Learning est une version étendue de l'algorithme traditionnel Q-Learning, qui emploie des réseaux neuronaux profonds pour estimer les valeurs Q.
Le Q-Learning classique est efficace dans les environnements qui ont un nombre limité et déterminé d'états, mais il éprouve des problèmes avec les espaces 
d'états étendus ou continus à cause de la taille de la table Q. Le Deep Q-Learning pallie cette contrainte en substituant la table Q à un réseau de neurones 
apte à estimer les valeurs Q pour chaque couple état-action.

## Installation et Utilisation
Pour visualiser l'apprentissage de DQN sur le gridword vous pouvez lancer ce projet.

### Étape 1 : cloner le projet
```bash
  git clone 
```

### Étape 2 : créer un environnement python
```bash
  python -m venv venv
  source ./venv/bin/activate
```

### Étape 3 : installer les dépendances
```bash
  pip install -r requirements.txt
```

### Étape 4 : lancer le projet
```bash
  python train_dqn.py
```
