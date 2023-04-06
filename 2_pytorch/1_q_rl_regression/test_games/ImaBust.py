import numpy as np
import pygame

# Dummy minesweeper-esk game (Ima Bust!):
# Win Condition: Turn all of the cells on the board to the max value
# Lose Condition: Selecting a cell that's already at the max value, causing it to flip back to min value (Bust!)
# Gameplay: Select 1 cell per turn
class ImaBust(object):
  def __init__(self, state_1_d_size, min_value, max_value, window_size, render_fps):
    self.window_size = window_size
    self.window = None
    self.clock = None
    self.render_fps = render_fps
    self.min_value = min_value
    self.max_value = max_value

    self.state_1_d_size = state_1_d_size
    
    self.restart()

  def restart(self):
    self.state = np.full((self.state_1_d_size * self.state_1_d_size), 1, dtype=np.int64)
    # self.state = np.random.randint(self.min_value, self.min_value + 3, (self.state_1_d_size * self.state_1_d_size))

  def apply_action(self, action=None):
    if action is not None:
      # increment selected cell and score
      self.state[action] += 1

      # Draw next game frame
      self._render_frame()

    return self.state

  def _render_frame(self):
    if self.window is None:
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.window_size, self.window_size)
      )
      self.font = pygame.font.Font('freesansbold.ttf', 32)
    if self.clock is None:
      self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))
    pix_square_size = round(
      self.window_size / self.state_1_d_size
    )  # The size of a single grid square in pixels

    text_list = []
    for i, cell in enumerate(self.state):
      text = self.font.render(str(cell.item()), True, (0, 255, 0) if cell.item() == self.max_value else (255, 0, 0))
      text_list.append(
        {
          "text": text,
          "pos": (((i % self.state_1_d_size) * pix_square_size) + (pix_square_size / 2) - 10,
                  ((i // self.state_1_d_size) * pix_square_size) + (pix_square_size / 2) - 10)
        }
      )

    # Finally, add some gridlines
    for x in range(self.state_1_d_size + 1):
      pygame.draw.line(
        canvas,
        0,
        (0, pix_square_size * x),
        (self.window_size, pix_square_size * x),
        width=3,
      )
      pygame.draw.line(
        canvas,
        0,
        (pix_square_size * x, 0),
        (pix_square_size * x, self.window_size),
        width=3,
      )

    # The following line copies our drawings from `canvas` to the visible window
    self.window.blit(canvas, canvas.get_rect())
    for text in text_list:
      self.window.blit(text["text"], text["pos"])
    pygame.event.pump()
    pygame.display.update()

    # We need to ensure that human-rendering occurs at the predefined framerate.
    # The following line will automatically add a delay to keep the framerate stable.
    self.clock.tick(self.render_fps)

