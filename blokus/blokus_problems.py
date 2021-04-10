from board import Board
from search import SearchProblem, ucs, astar, greedy_best_first
import util
import numpy as np


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = [[board_h - 1, 0], [0, 0], [0, board_w - 1], [board_h - 1, board_w - 1]]

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        Checks if the given state is the goal state
        """
        for target in self.targets:
            if state.get_position(target[1], target[0]) == -1:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        cost = 0
        for action in actions:
            cost += action.piece.num_tiles
        return cost


def blokus_corners_heuristic(state, problem):
    """
    this heuristic looks for the closest corner in the given state, afterwards it calculates
    the minimal number of tiles needed to be filled in order to complete fill all of the corners.
    """
    max_dist = max(state.board_w, state.board_h)
    min_dist = min(state.board_w, state.board_h)
    dists = []
    for target in problem.targets:
        if state.get_position(target[1], target[0]) != -1:
            continue
        dist = min_distance(target, state, max_dist)
        dists.append(dist)
    if len(dists) == 0:
        return 0
    return min(dists) + (len(dists) - 1) * (min_dist - 1)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        Checks if the given state is the goal state
        """
        for target in self.targets:
            if state.get_position(target[1], target[0]) == -1:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        cost = 0
        for action in actions:
            cost += action.piece.num_tiles
        return cost


def blokus_cover_heuristic(state, problem):
    """
    Heuristic for the covering problem.
    returns the minimal distance from one of the targets + number of free targets
    """
    max_dist = max(state.board_w, state.board_h)
    dists = []
    missing_targets = 0
    for target in problem.targets:
        if state.get_position(target[1], target[0]) == -1:  # target is free
            missing_targets += 1
            dists.append(
                min_distance(target, state, max_dist))  # add the minimal distance to that target at given state
    if len(dists) == 0:  # all targets are full
        return 0
    if max_dist + 1 in dists:  # board at an unsolvable state, return high cost
        return max_dist + 1 + missing_targets
    return min(dists) + missing_targets - 1  # minimal distance out of all targets + number of other free targets


def min_distance(target, state, max_dist):
    """
    Returns the minimal distance to given target from given board state
    """
    cords = np.where(state.state != -1)  # get all full squares
    min_dist = max_dist
    for i in range(len(cords[0])):
        man_dist = util.manhattanDistance((cords[0][i], cords[1][i]), target)  # man distance from square to target
        if man_dist == 1:  # unsolvable board
            return max_dist + 1
        if man_dist == 0:  # already found target
            return max_dist
        min_dist = min(man_dist - 1,
                       min_dist)  # to keep admissibility subtract 1 from man_dist since you can place pieces diagonally
    return min_dist


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        self.expanded = 0
        self.start = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.sec_boar = self.board.__copy__()
        self.targets_to_find = targets.copy()

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.sec_boar

    def is_goal_state(self, state):
        """
        Checks if the given state is the goal state
        """
        for target in self.targets:
            if state.get_position(target[1], target[0]) == -1:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action), where 'successor' is a
        successor to the current state and 'action' is the action
        required to get there.
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move) for move in state.get_legal_moves(0)]

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace
        return backtrace
        """
        # Run Greedy Best-First search with closest_heuristic to get solution
        moves = greedy_best_first(self, closest_heuristic)
        return moves


def get_number_of_missing_targets(state, targets):
    """
    returns the number of targets that are free in the given state of the board
    """
    n = 0
    for target in targets:
        if state.get_position(target[1], target[0]) == -1:
            n += 1
    return n


def closest_heuristic(state, p):
    """
    Inadmissible heuristic to find closest target at given state.
    returns the manhattan distance from closest target + number of free targets
    """
    closest_target, dist = find_closest_goal(state, p.targets)
    max_dist = max(state.board_w, state.board_h)
    # multiplying by max_dist makes us inadmissible but gives more weight to missing targets.
    missing_targets = get_number_of_missing_targets(state, p.targets) * max_dist
    if state.get_position(closest_target[1], closest_target[0]) != -1:
        return 0 + missing_targets
    return dist + missing_targets


def cheb_distance(c_1, c_2):
    """
    chebyshev distance, same as king distance in chess
    """
    dist = max(abs(c_1[0] - c_2[0]), abs(c_1[1] - c_2[1]))
    return dist


def find_closest_goal(state, targets):
    """
    Finds the closest target at given state based on chebyshev distance.
    returns both the target and the manhattan distance to it
    """
    cords = np.where(state.state != -1)  # find filled squares
    max_dist = max(state.board_h, state.board_w)
    dists = [max(state.board_h, state.board_w)] * len(targets)  # stores minimal distance for each target
    man_dists = [max(state.board_h, state.board_w)] * len(targets)
    for j, target in enumerate(targets):
        for i in range(len(cords[0])):
            cheb_dist = cheb_distance((cords[0][i], cords[1][i]), target)
            man_dist = util.manhattanDistance((cords[0][i], cords[1][i]), target)
            if cheb_dist == 0:  # target is filled, treat it as distant target so it's not chosen
                man_dists[j] = max_dist
                dists[j] = max_dist
                break
            if man_dist == 1:  # in case of manhattan distance of 1, the state leads to an unsolvable board
                man_dist = max_dist
            dists[j] = min(cheb_dist, dists[j])
            man_dists[j] = min(man_dist, man_dists[j])
    return targets[dists.index(min(dists))], man_dists[dists.index(min(dists))]


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """

        """
        for target in self.targets:
            if state.get_position(target[1], target[0]) == -1:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        cost = 0
        for action in actions:
            cost += action.piece.num_tiles
        return cost

    def solve(self):
        "*** YOUR CODE HERE ***"
        moves = astar(self, blokus_cover_heuristic)
        return moves
