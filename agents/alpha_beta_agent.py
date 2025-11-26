# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("alpha_beta_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
      
  def alpha_beta_pruning(self, chess_board, player, opponent, depth, max_player, alpha, beta):
    done, p1, p2 = check_endgame(chess_board)
    if depth == 0 or done:
      return p1 - p2 if max_player == 1 else p2 - p1
    all_moves = get_valid_moves(chess_board, player)
    if len(all_moves) == 0:
      return self.alpha_beta_pruning(chess_board, opponent, player, depth - 1, max_player, alpha, beta)
    
    if player == max_player:
      best_move = float('-inf')
      for i in all_moves:
        temp_board = deepcopy(chess_board)
        execute_move(temp_board, i, player)
        move_value = self.alpha_beta_pruning(temp_board, opponent, player, depth - 1, max_player, alpha, beta)
      
        if move_value > best_move:
          best_move = move_value
        alpha = max(alpha, best_move)

        if alpha >= beta:
          break
    else:
        best_move = float('inf')
        for i in all_moves:
          temp_board = deepcopy(chess_board)
          execute_move(temp_board, i, player)
          move_value = self.alpha_beta_pruning(temp_board, opponent, player, depth - 1, max_player, alpha, beta)
          if move_value < best_move:
            best_move = move_value
          beta = min(beta, best_move)
          if alpha >= beta:
            break

    return best_move

  def find_best_move(self, chess_board, player, opponent, depth):
    all_moves = get_valid_moves(chess_board, player)
    best_move = all_moves[0]
    best_score = float('-inf')
    max_player = player
    if len(all_moves) == 0: 
      return None
    for i in all_moves:
      temp_board = deepcopy(chess_board)
      execute_move(temp_board, i, player)

      move_value = self.alpha_beta_pruning(temp_board, opponent, player, depth - 1, max_player, float('-inf'), float('inf'))
      if move_value > best_score:
        best_score = move_value
        best_move = i
    return best_move 
    

    


  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "AlphaBetaAgent"
    self.max_depth = 2
    self.count = 0

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
    move = self.find_best_move(chess_board, player, opponent, self.max_depth)
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
  
    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return move

