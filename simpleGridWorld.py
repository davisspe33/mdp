import hiive.mdptoolbox 
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import numpy as np
import pandas as pd 

# grid world setup from 
# https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem

#(0,1,2,3)
#([a_up, a_down, a_left, a_right])
def main():
    p,r = grid_world_example()
    
    
    vi = hiive.mdptoolbox.mdp.ValueIteration(p, r, 0.9)
    x = vi.run()
    time = 0
    for i in x:
        time+= i["Time"] 
    print('\nValue Itteration:')
    print('value itteration optimal policy: ' + str(vi.policy))
    print('value itteration total time: ' + str(time))
    print('value itteration time to converge: '+ str(vi.time))
    print('value itteration ave time per itteration : ' + str(time/len(x)))
    print('value itteration # itterations : ' + str(len(x)))
    print('value itteration Max Value : ' + str(x[-1]['Max V']))

    print('\n\nPolicy Itteration:')
    pi = hiive.mdptoolbox.mdp.PolicyIteration(p ,r, 0.9)
    x= pi.run()
    time = 0
    for i in x:
        time+= i["Time"] 
    print('policy itteration optimal policy: ' + str(pi.policy))
    print('policy itteration total time: ' + str(time))
    print('policy itteration time to converge: '+ str(pi.time))
    print('policy itteration ave time per itteration : ' + str(time/len(x)))
    print('policy itteration # itterations : ' + str(len(x)))
    print('policy itteration Max Value : ' + str(x[-1]['Max V']))

    print('\n\nQ-learning:')
    q = hiive.mdptoolbox.mdp.QLearning(p ,r, 0.8, epsilon=.05)
    x = q.run()
    time = 0
    for i in x:
        time+= i["Time"] 
    print('q-learning optimal policy: ' + str(q.policy))
    print('q-learning total time: ' + str(time))
    print('q-learning time to converge: '+ str(q.time))
    print('q-learning ave time per itteration : ' + str(time/len(x)))
    print('q-learning # itterations : ' + str(len(x)))
    print('q-learning Max Value : ' + str(x[-1]['Max V']))
    
    print('\n')

def run(func):
    func()
    return func

def grid_world_example(grid_size=(3, 4),
                       black_cells=[(1,1)],
                       white_cell_reward=-0.02,
                       green_cell_loc=(0,3),
                       red_cell_loc=(1,3),
                       green_cell_reward=1.0,
                       red_cell_reward=-1.0,
                       action_lrfb_prob=(.1, .1, .8, 0.),
                       start_loc=(0, 0)
                      ):
    num_states = grid_size[0] * grid_size[1]
    num_actions = 4
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    @run
    def fill_in_probs():
        # helpers
        to_2d = lambda x: np.unravel_index(x, grid_size)
        to_1d = lambda x: np.ravel_multi_index(x, grid_size)

        def hit_wall(cell):
            if cell in black_cells:
                return True
            try: # ...good enough...
                to_1d(cell)
            except ValueError as e:
                return True
            return False

        # make probs for each action
        a_up = [action_lrfb_prob[i] for i in (0, 1, 2, 3)]
        a_down = [action_lrfb_prob[i] for i in (1, 0, 3, 2)]
        a_left = [action_lrfb_prob[i] for i in (2, 3, 1, 0)]
        a_right = [action_lrfb_prob[i] for i in (3, 2, 0, 1)]
        actions = [a_up, a_down, a_left, a_right]
        for i, a in enumerate(actions):
            actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}

        # work in terms of the 2d grid representation
     
        def update_P_and_R(cell, new_cell, a_index, a_prob):
            if cell == green_cell_loc:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = green_cell_reward

            elif cell == red_cell_loc:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = red_cell_reward

            elif hit_wall(new_cell):  # add prob to current cell
                P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                R[to_1d(cell), a_index] = white_cell_reward

            else:
                P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                R[to_1d(cell), a_index] = white_cell_reward

        for a_index, action in enumerate(actions):
            for cell in np.ndindex(grid_size):
                # up
                new_cell = (cell[0]-1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['up'])

                # down
                new_cell = (cell[0]+1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['down'])

                # left
                new_cell = (cell[0], cell[1]-1)
                update_P_and_R(cell, new_cell, a_index, action['left'])

                # right
                new_cell = (cell[0], cell[1]+1)
                update_P_and_R(cell, new_cell, a_index, action['right'])

    return P, R

main()