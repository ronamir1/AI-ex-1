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
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
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
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    cost = 0
    to_add = len(problem.targets) - 1
    max_dist = max(state.board_w, state.board_h)
    for target in problem.targets:
        if state.get_position(target[0], target[1]) != -1:
            to_add -= 1
            continue
        dist = min_distance(target, state, max_dist)
        cost = max(cost, dist)
    return cost + to_add


class BlokusCoverProblem(SearchProblem):
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
        "*** YOUR CODE HERE ***"
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
    cost = 0
    to_add = len(problem.targets) - 1
    max_dist = max(state.board_w, state.board_h)
    for target in problem.targets:
        if state.get_position(target[0], target[1]) != -1:
            to_add -= 1
            continue
        dist = min_distance(target, state, max_dist)
        cost = max(cost, dist)
    return cost


def min_distance(target, state, max_dist):
    cords = np.where(state.state != -1)
    min_dist = max_dist
    for i in range(len(cords[0])):
        man_dist = util.manhattanDistance((cords[0][i], cords[1][i]), target)
        if man_dist == 1:  ## unsolvable board
            return max_dist
        min_dist = min(man_dist - 1, min_dist)
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
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.sec_boar

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
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
        return [(state.do_move(0, move), move, move.piece.num_tiles
                 ) for move in state.get_legal_moves(0)]

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

        moves = greedy_best_first(self, closest_heuristic)
        return moves

        # q = util.PriorityQueue()
        # max_dist = max(cur_state.board_h, cur_state.board_w)
        # q.push(cur_state, 0)
        # prev = dict()
        # actions = []
        # prev[cur_state] = None
        # while not self.is_goal_state(cur_state):
        #     cur_state = q.pop()
        #     if self.is_goal_state(cur_state):
        #         while not cur_state == self.get_start_state():  # build actions list from goal state
        #             cur_state, cur_move = prev[cur_state]
        #             actions.append(cur_move)
        #         actions.reverse()
        #         return actions
        #     closest = self.find_closest_goal(cur_state)
        #     for suc in self.get_successors(cur_state):
        #         if suc[0] not in prev:
        #             prev[suc[0]] = (cur_state, suc[1])
        #             q.push(suc[0], self.get_score(suc, closest, max_dist))


def get_number_of_missing_targets(state, targets):
    n = 0
    for target in targets:
        if state.get_position(target[1], target[0]) == -1:
            n += 1
    return n


def closest_heuristic(state, p):
    closest_target = find_closest_goal(state, p.targets)
    max_dist = max(state.board_w, state.board_h)
    missing_targets = get_number_of_missing_targets(state, p.targets) * max_dist
    if state.get_position(closest_target[1], closest_target[0]) != -1:
        return 0 + missing_targets
    dist = min_distance(closest_target, state, max_dist)
    return dist + missing_targets

def cheb_distance(c_1, c_2):
    dist = max(abs(c_1[0] - c_2[0]), abs(c_1[1] - c_2[1]))
    return dist

def find_closest_goal(state, targets):
    cords = np.where(state.state != -1)
    max_dist = max(state.board_h, state.board_w)
    dists = [max(state.board_h, state.board_w)] * len(targets)
    for j, target in enumerate(targets):
        for i in range(len(cords[0])):
            man_dist = util.manhattanDistance((cords[0][i], cords[1][i]), target)
            cheb_dist = cheb_distance((cords[0][i], cords[1][i]), target)
            if man_dist == 0:
                dists[j] = max_dist
                break
            dists[j] = min(cheb_dist, dists[j])
    return targets[dists.index(min(dists))]


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
