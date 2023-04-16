import pygame
import numpy as np
# write tic tac toe game with pygame

# Create game screen
def create_game_screen():
  # Create game screen
  pygame.init()
  screen = pygame.display.set_mode((512, 512))
  pygame.display.set_caption("Tic Tac Toe")
  return screen

# Draw game board
def draw_game_board(screen):
  # Draw game board
  pygame.draw.line(screen, (255, 255, 255), (170, 0), (170, 512), 10)
  pygame.draw.line(screen, (255, 255, 255), (342, 0), (342, 512), 10)
  pygame.draw.line(screen, (255, 255, 255), (0, 170), (512, 170), 10)
  pygame.draw.line(screen, (255, 255, 255), (0, 342), (512, 342), 10)

# Draw game pieces
def draw_game_pieces(screen, game_state):
  # Draw game pieces
  for i in range(3):
    for j in range(3):
      if game_state[i][j] == 1:
        pygame.draw.circle(screen, (255, 255, 255), (85 + (i * 172), 85 + (j * 172)), 80, 10)
      elif game_state[i][j] == 2:
        pygame.draw.line(screen, (255, 255, 255), (25 + (i * 172), 25 + (j * 172)), (145 + (i * 172), 145 + (j * 172)), 10)
        pygame.draw.line(screen, (255, 255, 255), (145 + (i * 172), 25 + (j * 172)), (25 + (i * 172), 145 + (j * 172)), 10)

# Draw game screen
def draw_game_screen(screen, game_state):
  # Draw game screen
  screen.fill((0, 0, 0))
  draw_game_board(screen)
  draw_game_pieces(screen, game_state)
  pygame.display.update()

# Create game state
def create_game_state():
  # Create game state
  game_state = np.zeros((3, 3))
  return game_state

# Check if game is over
def check_game_over(game_state):
  # Check if game is over
  # Check for horizontal win
  for i in range(3):
    if game_state[i][0] == game_state[i][1] == game_state[i][2] and game_state[i][0] != 0:
      return True

  # Check for vertical win
  for i in range(3):
    if game_state[0][i] == game_state[1][i] == game_state[2][i] and game_state[0][i] != 0:
      return True

  # Check for diagonal win
  if game_state[0][0] == game_state[1][1] == game_state[2][2] and game_state[0][0] != 0:
    return True
  if game_state[2][0] == game_state[1][1] == game_state[0][2] and game_state[2][0] != 0:
    return True

  # Check for draw
  for i in range(3):
    for j in range(3):
      if game_state[i][j] == 0:
        return False

  return True

# Run game
def run_game():
  # Run game
  # Create game screen
  screen = create_game_screen()

  # Create game state
  game_state = create_game_state()

  # Draw game screen
  draw_game_screen(screen, game_state)

  # Run game loop
  running = True
  player = 1
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

      if event.type == pygame.MOUSEBUTTONDOWN:
        x = event.pos[0]
        y = event.pos[1]
        if x < 170:
          i = 0
        elif x < 342:
          i = 1
        else:
          i = 2
        if y < 170:
          j = 0
        elif y < 342:
          j = 1
        else:
          j = 2
        if game_state[i][j] == 0:
          game_state[i][j] = player
          if player == 1:
            player = 2
          else:
            player = 1
          draw_game_screen(screen, game_state)
          if check_game_over(game_state):
            running = False

  pygame.quit()

run_game()