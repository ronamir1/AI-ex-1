"""
In search.py, you will implement generic search algorithms
"""

import util
import numpy as np


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

	print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    fringe = util.Stack()
    return search(problem, fringe)


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    fringe = util.Queue()
    return search(problem, fringe)



def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    actions = []
    prev = dict()  # son state to father state
    costs = dict()
    fringe = util.PriorityQueue()
    fringe.push(problem.get_start_state(), 0)
    costs[problem.get_start_state()] = 0
    while not fringe.isEmpty():
        cur_state = fringe.pop()
        if problem.is_goal_state(cur_state):
            while not cur_state == problem.get_start_state():  # build actions list from goal state
                cur_state, cur_move = prev[cur_state]
                actions.append(cur_move)
            actions.reverse()
            return actions
        for suc in problem.get_successors(cur_state):  # expand graph
            if suc[0] not in prev:
                prev[suc[0]] = (cur_state, suc[1])
                cost = costs[cur_state] + suc[2]
                costs[suc[0]] = cost
                fringe.push(suc[0], cost)



def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    actions = []
    prev = dict()  # son state to father state
    costs = dict()
    fringe = util.PriorityQueue()
    fringe.push(problem.get_start_state(), 0)
    costs[problem.get_start_state()] = 0
    prev[problem.get_start_state()] = None
    while not fringe.isEmpty():
        cur_state = fringe.pop()
        if problem.is_goal_state(cur_state):
            while not cur_state == problem.get_start_state():  # build actions list from goal state
                cur_state, cur_move = prev[cur_state]
                actions.append(cur_move)
            actions.reverse()
            return actions
        for suc in problem.get_successors(cur_state):  # expand graph
            if suc[0] not in prev:
                cost = costs[cur_state] + suc[2]
                costs[suc[0]] = cost
                fringe.push(suc[0], cost + heuristic(suc[0], problem))
                prev[suc[0]] = (cur_state, suc[1])
    return actions


def search(problem, fringe):
    actions = []
    prev = dict()  # son state to father state
    start_state = problem.get_start_state()
    fringe.push(start_state)
    prev[start_state] = None
    while not fringe.isEmpty():
        cur_state = fringe.pop()
        if problem.is_goal_state(cur_state):
            while not cur_state == start_state:  # build actions list from goal state
                cur_state, cur_move = prev[cur_state]
                actions.append(cur_move)
            actions.reverse()
            return actions
        for suc in problem.get_successors(cur_state):  # expand graph
            if suc[0] not in prev:
                prev[suc[0]] = (cur_state, suc[1])
                fringe.push(suc[0])




# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
