from dqn.replay_buffer import *


class ClassificationEnv:
    def __init__(self, X, y):  # Ajout de 'y' ici
        self.X = X
        self.y = y
        self.state_size = X.shape[1]  # Devrait être 2 (pos_x, pos_y) [cite: 17]
        self.action_size = 4  # 4 actions possibles (Haut, Bas, Gauche, Droite) [cite: 17]
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.X[self.current_step]

    def step(self, action):
        # Récupérer l'action optimale attendue (label)
        correct_action = self.y[self.current_step]

        # Système de récompense : +1 si correct, -1 sinon [cite: 30]
        reward = 1.0 if action == correct_action else -1.0

        self.current_step += 1
        done = self.current_step >= len(self.X) - 1

        if not done:
            next_state = self.X[self.current_step]
        else:
            next_state = np.zeros(self.state_size)

        return next_state, reward, done
