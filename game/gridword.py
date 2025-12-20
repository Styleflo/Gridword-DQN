from collections import deque
from random import randint
import numpy as np
import pygame
import sys

class GridWorld5x5:
    def __init__(self, max_steps=float('inf')):
        self.success = False
        self.max_steps = max_steps
        self.steps = 0

        self.grid_size = 5
        self.cell_size = 700 // self.grid_size
        self.window_size = self.grid_size * self.cell_size

        self.start_pos = (0, 0)
        self.goal_pos = (self.grid_size -1, self.grid_size -1)
        self.obstacles = None
        self.generateObstacles()

        self.action_space = 4
        self.state_space = self.grid_size * self.grid_size

        self.state_size = self.grid_size * self.grid_size  # 25
        self.action_size = self.action_space

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size)
        )
        pygame.display.set_caption("GridWorld 5x5")

        self.clock = pygame.time.Clock()
        self.reset()

    def generateObstacles(self):
        nb_obstacles = (self.grid_size ** 2) * 20 // 100

        def path_exists(obstacles_set):
            visited = set()
            queue = deque([self.start_pos])
            visited.add(self.start_pos)

            while queue:
                x, y = queue.popleft()

                if (x, y) == self.goal_pos:
                    return True

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy

                    if (
                            0 <= nx < self.grid_size and
                            0 <= ny < self.grid_size and
                            (nx, ny) not in obstacles_set and
                            (nx, ny) not in visited
                    ):
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            return False

        # Génération jusqu'à avoir un chemin valide
        while True:
            obstacles = set()

            while len(obstacles) < nb_obstacles:
                x = randint(0, self.grid_size - 1)
                y = randint(0, self.grid_size - 1)
                pos = (x, y)

                if pos != self.start_pos and pos != self.goal_pos:
                    obstacles.add(pos)

            if path_exists(obstacles):
                self.obstacles = list(obstacles)
                break

    def reset(self):
        self.success = False
        self.steps = 0
        x = randint(0, self.grid_size - 1)
        y = randint(0, self.grid_size - 1)
        self.start_pos = (x, y)
        self.agent_pos = self.start_pos
        self.generateObstacles()
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        x, y = self.agent_pos
        state[x * self.grid_size + y] = 1.0

        for (x, y) in self.obstacles:
            state[x * self.grid_size + y] = -1.0

        ax, ay = self.goal_pos
        state[ax * self.grid_size + ay] = 2.0
        return state

    def step(self, action):
        x, y = self.agent_pos
        self.steps += 1

        old_pos = (x,y)
        gx, gy = self.goal_pos
        old_dist = abs(x - gx) + abs(y - gy)

        if action == 0:      # Haut
            x -= 1
        elif action == 1:    # Bas
            x += 1
        elif action == 2:    # Gauche
            y -= 1
        elif action == 3:    # Droite
            y += 1

        # Limites
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            x, y = self.agent_pos

        new_pos = (x, y)
        new_dist = abs(x - gx) + abs(y - gy)
        reward = -0.5

        if new_pos in self.obstacles:
            reward += -10
            done = True
        elif new_pos == self.goal_pos:
            reward += 200
            self.success = True
            done = True

        else:
            reward += 3 * (old_dist - new_dist)
            done = False

        if self.steps >= self.max_steps:
            done = True

        self.agent_pos = new_pos
        return self._get_state(), reward, done

    def render_pygame(self):
        self.screen.fill((255, 255, 255))

        # Grille
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Obstacles
        for (x, y) in self.obstacles:
            rect = pygame.Rect(
                y * self.cell_size + 2,
                x * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size -4
            )
            pygame.draw.rect(self.screen, (200, 0, 0), rect)

        # Objectif
        gx, gy = self.goal_pos
        rect = pygame.Rect(
            gy * self.cell_size + 2,
            gx * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, (0, 200, 0), rect)

        # Agent
        ax, ay = self.agent_pos
        center = (
            ay * self.cell_size + self.cell_size // 2,
            ax * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(
            self.screen,
            (0, 0, 200),
            center,
            self.cell_size // 3
        )

        pygame.display.flip()
        self.clock.tick(10)  # FPS

    def close(self):
        pygame.quit()
        sys.exit()