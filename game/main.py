from game.gridword import *

env = GridWorld5x5()
state = env.reset()

running = True
while running:
    env.render_pygame()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 0
            elif event.key == pygame.K_DOWN:
                action = 1
            elif event.key == pygame.K_LEFT:
                action = 2
            elif event.key == pygame.K_RIGHT:
                action = 3
            else:
                continue

            state, reward, done = env.step(action)

            if done:
                state = env.reset()

env.close()
