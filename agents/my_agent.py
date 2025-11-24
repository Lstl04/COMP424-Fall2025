# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, \
get_valid_moves, check_move_validity, MoveCoordinates, get_directions, \
get_two_tile_directions

@register_agent("my_agent")
class MyAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self, b=200):
    super(MyAgent, self).__init__()
    self.name = "MyAgent"
    self.tree = set()
    self.Q_dict = dict()
    self.N_dict = dict()
    self.Q_bar_dict = dict()
    self.N_bar_dict = dict()
    # Constant bias value (tune this parameter!)
    self.b = b

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    sim_board = deepcopy(chess_board)
    while time.time() - start_time < 1.9:
      self.sim(sim_board, player)
    
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")
    return self.tree_policy(chess_board, player)

  def default_policy(self, board, player):
    # The random move function requires all valid moves to be found (INEFFICIENT)
    # Here we choose the first valid move found, speeding up the default policy
    board_size = 7
    for r in range(board_size):
      for c in range(board_size):
        # Check square has a player's disc
        if board[r, c] == player:
          src = (r,c)
          # loop over all possible moves
          candidate_move_list = get_directions()
          candidate_move_list.extend(get_two_tile_directions())

          for direc in candidate_move_list:
            dest_tile = (r + direc[0], c + direc[1])
            valid_move = MoveCoordinates(src=(r,c), dest=dest_tile)
            if check_move_validity(board, valid_move, player):
              return valid_move
    # No valid moves
    return None

  def rollout(self, board, player):
    moves = []
    while not check_endgame(board)[0]:
      move = self.default_policy(board, player)
      execute_move(board, move, player)
      moves.append(move)

      if player == 1:
        player = 2
      else:
        player = 1
    
    p1 = np.count_nonzero(board == 1)
    p2 = np.count_nonzero(board == 2)
    
    return (moves, p1 - p2)

  def selection(self, board, player):
    board_move_pairs = []
    while check_endgame(board)[0]:
      board_key = board.to_bytes()
      if board_key not in self.tree:
        self.new_node(board, player)
        move = self.default_policy(board, player)
        board_move_pairs.append((deepcopy(board), move))
        return board_move_pairs
      
      move = self.tree_policy(board, player)
      execute_move(board, move, player)
      board_move_pairs.append((board, move))
      
      # Switch player
      if player == 1:
        player = 2
      else:
        player = 1
    
    return board_move_pairs

  def heuristic(self, board, move, player):
    sim_board = deepcopy(board)
    execute_move(sim_board, move, player)

    p1 = np.count_nonzero(sim_board == 1)
    p2 = np.count_nonzero(sim_board == 2)
    q = p1 - p2
    # obs = np.count_nonzero(board == 3)
    # empty_squares = 149 - (p1 + p2 + obs)

    # if player == 1:
    #   q = p1 + empty_squares // 2
    # else:
    #   q = p2 + empty_squares // 2

    return q, 1, q, 1

  def new_node(self, board, player):
    # Cannot hash a numpy ndarray, so use raw bytes as key
    board_key = board.tobytes()
    self.tree.add(board_key)

    board_size = 7
    for r in range(board_size):
      for c in range(board_size):
        # Check square has a player's disc
        if board[r, c] == player:
          src = (r,c)
          # loop over all possible moves
          candidate_move_list = get_directions()
          candidate_move_list.extend(get_two_tile_directions())

          for direc in candidate_move_list:
            dest_tile = (r + direc[0], c + direc[1])
            valid_move = MoveCoordinates(src=(r,c), dest=dest_tile)
            if check_move_validity(board, valid_move, player):
              q, n, q_bar, n_bar = self.heuristic(board, valid_move, player)
              self.Q_dict[(board_key, valid_move)] = q
              self.N_dict[(board_key, valid_move)] = n
              self.Q_bar_dict[(board_key, valid_move)] = q_bar
              self.N_bar_dict[(board_key, valid_move)] = n_bar

  def tree_policy(self, board, player):
    argmax_move = (None, float("-inf"))
    argmin_move = (None, float("inf"))

    board_size = 7
    for r in range(board_size):
      for c in range(board_size):
        # Check square has a player's disc
        if board[r, c] == player:
          src = (r,c)
          # loop over all possible moves
          candidate_move_list = get_directions()
          candidate_move_list.extend(get_two_tile_directions())

          for direc in candidate_move_list:
            dest_tile = (r + direc[0], c + direc[1])
            valid_move = MoveCoordinates(src=(r,c), dest=dest_tile)
            if check_move_validity(board, valid_move, player):
              evaluation = self.evaluate(board, valid_move)
              if evaluation > argmax_move[1]:
                argmax_move = (valid_move, evaluation)
              if evaluation < argmin_move[1]:
                argmin_move = (valid_move, evaluation)

    if player == 1:
      return argmax_move[0]
    else:
      return argmin_move[0]


  def evaluate(self, board, move):
    board_key = board.tobytes()

    # Use safe defaults so we don't KeyError, although the only case where we should
    # be getting a KeyError is if we do not have the time to add the board we start
    # the search at to the tree (which would be really bad)
    N = self.N_dict.get((board_key, move), 0)
    N_bar = self.N_bar_dict.get((board_key, move), 0)
    Q = self.Q_dict.get((board_key, move), 0.0)
    Q_bar = self.Q_bar_dict.get((board_key, move), 0.0)

    beta_denom = N + N_bar + 4 * N * N_bar * (self.b ** 2)
    beta = (N_bar / beta_denom) if beta_denom != 0 else 0
    return (1 - beta) * Q + beta * Q_bar

  def backup(self, board_move_pairs, default_moves, z):
    all_moves = [pair[1] for pair in board_move_pairs] + default_moves
    D = len(all_moves)
    for i, (board, move) in enumerate(board_move_pairs):
      board_key = board.tobytes()
      self.N_dict[(board_key, move)] = (
        self.N_dict.get((board_key, move), 0) + 1
      )
      # We don't need to worry about N_dict being 0 as we just incremented it by 1
      self.Q_dict[(board_key, move)] = (
        self.Q_dict.get((board_key, move), 0.0) + 
        (z - self.Q_dict.get((board_key, move), 0.0)) / self.N_dict.get[(board_key, move)]
      )
      moves_by_player = set()
      for j in range(i, D, 2):
        moves_by_player.add(all_moves[i])
      for k in range(i, D, 2):
        move_u = all_moves[k]
        if move_u not in moves_by_player:
          self.N_bar_dict[(board_key, move_u)] = (
            self.N_bar_dict.get((board_key, move), 0) + 1
          )
          # We don't need to worry about N_bar_dict being 0 as we just incremented it by 1
          self.Q_bar_dict[(board_key, move_u)] = (
            (z - self.Q_bar_dict.get((board_key, move), 0.0)) / self.N_bar_dict[(board_key, move)]
          )

  def sim(self, board, player):
    board_move_pairs = self.selection(board, player)
    default_moves, z = self.rollout(board, player)
    self.backup(board_move_pairs, default_moves, z)
