from game.gridword import *
from dqn.dqn_agent import *

import pygame
import torch
import time

env = GridWorld5x5(max_steps=1000)
state = env.reset()

done = False

# print("=== DÉBUT DE L'ENTRAINEMENT DQN ===")
# rewards = train_dqn(env, episodes=5000)
# print("=== FIN DE L'ENTRAINEMENT DQN ===")

# ==============================
# 1. INIT PYGAME
# ==============================
pygame.init()

# IMPORTANT : création de la fenêtre AVANT le jeu
screen_size = 500
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("DQN - GridWorld 5x5")

clock = pygame.time.Clock()

# ==============================
# 2. ENVIRONNEMENT
# ==============================
env = GridWorld5x5()
env.screen = screen  # connecter l'écran Pygame

state_size = env.state_size      # 25
action_size = env.action_size    # 4

# ==============================
# 3. AGENT DQN
# ==============================
agent = DQNAgent(state_size, action_size)

# Pas d'exploration en test
agent.epsilon = 0.0

# ==============================
# 4. CHARGEMENT DU MODÈLE
# ==============================
model_path = "models/dqn_gridword.pth"
device = torch.device("cpu")
agent.device = device
agent.q_network.to(device)

try:
    agent.q_network.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    agent.q_network.eval()
    print("✅ Modèle DQN chargé avec succès")
except FileNotFoundError:
    print(f"❌ Modèle introuvable : {model_path}")
    pygame.quit()
    raise SystemExit

# ==============================
# 5. BOUCLE DE JEU
# ==============================
running = True
state = env.reset()
env.render_pygame()
time.sleep(1)
done = False
total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not done:
        action = agent.act(state)
        time.sleep(0.5)
        print("Action choisie :", action)
        state, reward, done = env.step(action)
        total_reward += reward

        env.render_pygame()
        clock.tick(2)  # vitesse d'observation

    else:
        # Episode terminé → pause visuelle
        env.render_pygame()
        clock.tick(5)

        # Attente de 3 secondes
        time.sleep(3)

        # Réinitialisation de l'environnement
        state = env.reset()
        env.render_pygame()
        done = False
        total_reward = 0

# ==============================
# 6. FIN
# ==============================
pygame.quit()
print(f"🏁 Score final : {total_reward:.2f}")