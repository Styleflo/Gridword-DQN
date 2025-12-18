import pygame
import sys
import numpy as np

class GridWorld5x5:
    def __init__(self):
        self.success = False
        self.grid_size = 5
        self.cell_size = 100
        self.window_size = self.grid_size * self.cell_size

        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 1), (2, 4), (3, 2)]

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

    def reset(self):
        self.success = False
        self.agent_pos = self.start_pos
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        x, y = self.agent_pos
        state[x * self.grid_size + y] = 1.0
        return state

    def step(self, action):
        x, y = self.agent_pos

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

        if new_pos in self.obstacles:
            reward = -10
            done = True
        elif new_pos == self.goal_pos:
            reward = 10
            self.success = True
            done = True
        else:
            reward = -0.1
            done = False

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
                y * self.cell_size,
                x * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (200, 0, 0), rect)

        # Objectif
        gx, gy = self.goal_pos
        rect = pygame.Rect(
            gy * self.cell_size,
            gx * self.cell_size,
            self.cell_size,
            self.cell_size
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