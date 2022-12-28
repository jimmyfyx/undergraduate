import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0   # current points before state update 
        self.s = None     # at state s_t, this is s_t-1
        self.a = None     # at state s_t, this is the action taken at s_t-1 that results in s_t
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO: write your function here
        # Time step t, at state s_t = s_prime
        if self._train == True:
            # training mode
            if self.s != None and self.a != None:
                # calculate the highest Q value at s_t among all the actions a at time step t
                max_Q = -10000
                for action in self.actions:
                    Q = self.Q[s_prime][action]
                    if Q > max_Q:
                        max_Q = Q
                
                # calculate the reward at time step t
                if points > self.points:
                    reward = 1
                elif dead == True:
                    reward = -1
                else:
                    reward = -0.1
                
                # apply the Q update formula
                self.N[self.s][self.a] += 1
                lr = self.C / (self.C + self.N[self.s][self.a])
                self.Q[self.s][self.a] = self.Q[self.s][self.a] + lr * (reward + self.gamma * max_Q - self.Q[self.s][self.a])

            # determine the next action depending on whether the agent is dead
            if dead == False:
                # update bookkeeping variables
                self.points = points
                self.s = s_prime

                # exploration or exploitation
                best_action = 0
                f_QN = -10000
                for action in self.actions:
                    N_s_a = self.N[s_prime][action]
                    if N_s_a < self.Ne:
                        if f_QN <= 1:
                            f_QN = 1
                            best_action = action
                    else:
                        if self.Q[s_prime][action] >= f_QN:
                            f_QN = self.Q[s_prime][action]
                            best_action = action
                
                self.a = best_action
                return best_action
            else:
                # the agent is dead, action does not matter
                self.reset()
                return 0
        else:
            # testing mode
            # choose the action with the highest Q value
            max_Q = -10000
            best_action = 0
            for action in self.actions:
                Q = self.Q[s_prime][action]
                if Q > max_Q:
                    max_Q = Q
                    best_action = action
            return best_action


    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]

        # determine food direction
        if snake_head_x == food_x:
            food_dir_x = 0
        elif food_x < snake_head_x:
            food_dir_x = 1
        else:
            food_dir_x = 2
        
        if snake_head_y == food_y:
            food_dir_y = 0
        elif food_y < snake_head_y:
            food_dir_y = 1
        else:
            food_dir_y = 2
        
        # determine adjoining wall
        if snake_head_x - 1 != 0 and snake_head_x + 1 != utils.DISPLAY_WIDTH - 1:
            adjoining_wall_x = 0
        elif snake_head_x - 1 == 0:
            adjoining_wall_x = 1
        else:
            adjoining_wall_x = 2
        
        if snake_head_y + 1 != utils.DISPLAY_HEIGHT - 1 and snake_head_y - 1 != 0:
            adjoining_wall_y = 0
        elif snake_head_y + 1 == utils.DISPLAY_HEIGHT - 1:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 1
        
        # determine adjoining body
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        for body_pos in snake_body:
            if snake_head_y - 1 == body_pos[1] and snake_head_x == body_pos[0]:
                adjoining_body_top = 1
            if snake_head_y + 1 == body_pos[1] and snake_head_x == body_pos[0]:
                adjoining_body_bottom = 1
            if snake_head_x - 1 == body_pos[0] and snake_head_y == body_pos[1]:
                adjoining_body_left = 1
            if snake_head_x + 1 == body_pos[0] and snake_head_y == body_pos[1]:
                adjoining_body_right = 1

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)