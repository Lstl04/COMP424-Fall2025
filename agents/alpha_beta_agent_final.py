# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves
from collections import OrderedDict


@register_agent("alpha_beta_agent_final")
class AlphaBetaAgentFinal(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def board_key(self, board):
        """
        Create a unique key for the board state using its byte representation.
        This helps in identifying identical board states efficiently.
        """
        array = np.ascontiguousarray(board)
        return array.tobytes()

    def get_board_id(self, board, player):
        """
        Create or retrieve a unique ID for the given board state and player.
        This ID is used for caching and transposition table lookups.
        """
        # Two boards would need different ID's if it is a different player moving from there
        key = (self.board_key(board), player)
        # If no id has been generated for this board yet, generate one
        if key not in self.board_id_map:
            self.board_id_map[key] = self.next_board_id
            self.next_board_id += 1

        # Return the boards corresponding id
        return self.board_id_map[key]


    def alpha_beta_pruning(
        self, chess_board, player, opponent, depth, max_player, alpha, beta
    ):
        """
        Recursive alpha-beta pruning function.
        """
        # Raise TimeoutError when we pass the time limit
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()

        # Return the current score of the game if depth is 0 or game over
        done, p1, p2 = check_endgame(chess_board)
        if depth == 0 or done:
            return p1 - p2 if max_player == 1 else p2 - p1
        board_id = self.get_board_id(chess_board, player)

        # Load the list of moves from this board from cache if available
        # Load the list of move to the cache if not already available
        if board_id in self.moves:
            # Move the used board_id to the end to mark it recently used
            self.moves.move_to_end(board_id)
            all_moves = self.moves[board_id]

        else:
            move_list = get_valid_moves(chess_board, player)
            self.moves[board_id] = move_list
            all_moves = move_list
            # Drop the least recently used board from the moves cache if the cache is too large
            if len(self.moves) > self.max_moves_cache:
                self.moves.popitem(last=False)

        # If there are no valid moves, continue the search from the opponent's perspective
        if len(all_moves) == 0:
            return self.alpha_beta_pruning(
                chess_board, opponent, player, depth - 1, max_player, alpha, beta
            )
        
        move_scores = dict()

        if player == max_player:
            # Alpha value for this node
            best_move = float("-inf")
            for i in all_moves:
                # Exit if we exceed the time-limit
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                
                # Excecute the move, and continue search from the opponent's perspective
                temp_board = chess_board.copy()
                execute_move(temp_board, i, player)
                move_value = self.alpha_beta_pruning(
                    temp_board, opponent, player, depth - 1, max_player, alpha, beta
                )

                move_scores[(i.get_src(), i.get_dest())] = move_value

                # Keep track of the best move found so far (from the max-player's perspective),
                # and update alpha for this node if we found a better move
                if move_value > best_move:
                    best_move = move_value
                alpha = max(alpha, best_move)

                # Break out of the loop if we have an inconsistency
                if alpha >= beta:
                    break
        else:
            # Beta value for this node
            best_move = float("inf")
            # If all_moves is ordered, it is in decreasing order.
            # As the min-player, we want to search the moves with the lowest
            # score first
            for i in reversed(all_moves):
                # Exit if we exceed the time-limit
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                
                # Excecute the move, and continue search from the opponent's perspective
                temp_board = chess_board.copy()
                execute_move(temp_board, i, player)
                move_value = self.alpha_beta_pruning(
                    temp_board, opponent, player, depth - 1, max_player, alpha, beta
                )

                move_scores[(i.get_src(), i.get_dest())] = move_value

                # Keep track of the best move found so far (from the min-player's perspective),
                # and update beta for this node if we found a better move
                if move_value < best_move:
                    best_move = move_value
                beta = min(beta, best_move)

                # Break out of the loop if we have an inconsistency
                if alpha >= beta:
                    break

        # Only sort each depth once. Further sorting gives diminishing returns for the cost of sorting
        if depth == 1:
            self.moves[board_id] = sorted(all_moves, key=lambda x: move_scores.get((x.get_src(), x.get_dest()), 0), reverse=True)
        return best_move

    def find_best_move(self, chess_board, player, opponent, depth, alpha, beta):
        board_id = self.get_board_id(chess_board, player)

        # Load the list of moves from this board from cache if available
        # Load the list of move to the cache if not already available
        if board_id in self.moves:
            # Move the used board_id to the end to mark it recently used
            self.moves.move_to_end(board_id)
            all_moves = self.moves[board_id]
        
        else:
            move_list = get_valid_moves(chess_board, player)
            self.moves[board_id] = move_list
            all_moves = move_list
            # Drop the least recently used board from the moves cache if the cache is too large
            if len(self.moves) > self.max_moves_cache:
                self.moves.popitem(last=False)

        # If no moves can be made, return None for the move, and the previous score for the score
        if len(all_moves) == 0:
            return None, self.prev_score

        # Initialize best move and score
        best_move = all_moves[0]
        best_score = float("-inf")
        max_player = player

        # Evaluate each move from the root and record the scores to sort them
        move_scores = dict()

        for mv in all_moves:
            # Exit if we exceed the time-limit
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
            
            # Excecute the move, and continue the search from the opponent's perspective
            temp_board = chess_board.copy()
            execute_move(temp_board, mv, player)

            # Note that we use the alpha and beta values provided to this function,
            # as we do aspiration search
            move_value = self.alpha_beta_pruning(
                temp_board,
                opponent,
                player,
                depth - 1,
                max_player,
                alpha,
                beta,
            )

            # Convert the move to a tuple for hashing, and store the move's score
            move_scores[(mv.get_src(), mv.get_dest())] = move_value
            
            # Update best move if we found a better one
            if move_value > best_score:
                best_score = move_value
                best_move = mv

            # Break if our original estimate of the best possible score is exceeded
            # (Aspiration search)  
            if best_score >= beta:
                break

        # Sort the moves for additional depth iterations, and possible future revisits.
        self.moves[board_id] = sorted(all_moves, key=lambda x: move_scores.get((x.get_src(), x.get_dest()), 0), reverse=True)
      
        return best_move, best_score

    def __init__(self):
        super(AlphaBetaAgentFinal, self).__init__()
        self.name = "AlphaBetaAgentFinal"
        # Don't really need to set a max depth since we use iterative deepening and terminate
        # based on the time, but it is not like we will reach this depth anyway
        self.max_depth = 1000
        self.count = 0
        # caches, some with LRU (least recently used) eviction
        # Inspired by COMP 310 course at McGill
        self.moves = OrderedDict()
        self.board_id_map = dict()
        self.next_board_id = 0
        # Maximum time per move
        self.time_limit = 1.99
        # Size limits for caches
        self.max_moves_cache = 20000

        # Aspiration window initial delta, and previous score to
        # initialize the window around.
        # We start delta at 2, as at the beginning of the game,
        # the score cannot be changed by more than 1
        self.delta = 2
        self.prev_score = 0

    def step(self, chess_board, player, opponent):
        self.start_time = time.time()
        # Initialize best move and score, and the last completed depth
        best_move = None
        best_score = self.prev_score
        last_completed_depth = 0

        # Iterative deepening search
        for depth in range(2, self.max_depth + 1):
            # Exit if we exceed the time-limit
            if time.time() - self.start_time > self.time_limit:
                break

            # Set the aspiration window. Our scores are bounded between -49 and 49 (inclusive)
            alpha = max(-49, self.prev_score - self.delta)
            beta = min(49, self.prev_score + self.delta)

            # Initialize candidate move and score
            candidate = None
            candidate_score = self.prev_score

            # Use try-except to exit when we exceed the time-limit
            try:
                candidate, candidate_score = self.find_best_move(chess_board, player, opponent, depth, alpha, beta)
            except Exception:
                break

            # If the candidate score is not inside the aspiration window, we need to
            # widen the window and try again. 
            # We double the delta until we get a score within the window.
            # Once again, the try-except is used to exit when we exceed the time-limit
            try:
                while candidate_score <= alpha or candidate_score >= beta:
                  # Increase the aspiration window size by 2, and retry
                  self.delta += 2
                  alpha = max(-49, self.prev_score - self.delta)
                  beta = max(49, self.prev_score + self.delta)
                  candidate, candidate_score = self.find_best_move(chess_board, player, opponent, depth, alpha, beta)
            except Exception:
                break
            
            # We never need delta to be greater than 10, as no turn can
            # change the score by more than 9.
            # We reset delta to 7, as a smaller delta is better for aspiration search,
            # and if we are playing against strong opponents, it is highly unlikely for us
            # to affect the score by more than 7 in a single turn.
            if self.delta > 7:
                self.delta = 7

            # If we found a candidate move within the aspiration window, update best move and score,
            # as deeper searches are more accurate
            if candidate is not None:
                best_move = candidate
                best_score = candidate_score
                last_completed_depth = depth

        # Update delta for next turn's aspiration window
        self.prev_score = best_score

        time_taken = time.time() - self.start_time
        print("My AI's turn took ", time_taken, "seconds. Depth:", last_completed_depth)

        return best_move
