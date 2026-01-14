import pygame
import random
import numpy as np
import time

# ================== GAME SETTINGS ==================
WIDTH, HEIGHT = 400, 600
LANES = [100, 200, 300]
PLAYER_Y = 520
OBSTACLE_SPEED = 6

# ================== RL SETTINGS ==================
ACTIONS = [-1, 0, 1]  # LEFT, STAY, RIGHT
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05

# State: (left_sensor, front_sensor, right_sensor)
Q = np.zeros((2, 2, 2, 3))

# ================== INIT ==================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Subway RL with Sensors (Real Learning)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 26)

# ================== RESET ==================
def reset():
    player_lane = 1
    obstacles = [(random.randint(0, 2), -50)]
    return player_lane, obstacles

player_lane, obstacles = reset()
episode = 0
score = 0

# ================== SENSOR FUNCTION ==================
def get_sensors(player_lane, obstacles):
    left = front = right = 1

    for lane, y in obstacles:
        if y > PLAYER_Y - 120:
            if lane == player_lane:
                front = 0
            elif lane == player_lane - 1:
                left = 0
            elif lane == player_lane + 1:
                right = 0

    return left, front, right

# ================== MAIN LOOP ==================
running = True
while running:
    clock.tick(60)
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ================== SENSORS ==================
    state = get_sensors(player_lane, obstacles)

    # ================== ACTION ==================
    if random.random() < EPSILON:
        action = random.randint(0, 2)
    else:
        action = np.argmax(Q[state])

    move = ACTIONS[action]
    player_lane = max(0, min(2, player_lane + move))

    # ================== UPDATE OBSTACLES ==================
    reward = 1
    done = False

    new_obstacles = []
    for lane, y in obstacles:
        y += OBSTACLE_SPEED
        if y > PLAYER_Y:
            if lane == player_lane:
                reward = -100
                done = True
            else:
                score += 1
        else:
            new_obstacles.append((lane, y))

    obstacles = new_obstacles

    if random.random() < 0.03:
        obstacles.append((random.randint(0, 2), -50))

    # ================== NEXT STATE ==================
    next_state = get_sensors(player_lane, obstacles)

    # ================== Q UPDATE ==================
    Q[state][action] += ALPHA * (
        reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
    )

    # ================== DRAW ==================
    pygame.draw.rect(
        screen, (0, 200, 0),
        (LANES[player_lane] - 25, PLAYER_Y, 50, 50)
    )

    for lane, y in obstacles:
        pygame.draw.rect(
            screen, (200, 0, 0),
            (LANES[lane] - 25, y, 50, 50)
        )

    text = font.render(
        f"Episode: {episode}  Score: {score}  ε: {EPSILON:.2f}",
        True, (255, 255, 255)
    )
    screen.blit(text, (10, 10))

    pygame.display.flip()

    # ================== RESET ==================
    if done:
        episode += 1
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        time.sleep(0.3)
        player_lane, obstacles = reset()
        score = 0

pygame.quit()
