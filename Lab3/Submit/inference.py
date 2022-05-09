import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # ================================ Forward messages ================================
    # initialization 
    alpha_z0_dic = {}
    Observe_0 = observations[0]
    P_O0_given_Z0 = 1
    P_z0 = None

    #all posible start pos
    for z0 in all_possible_hidden_states:
        if Observe_0 != None:
            P_O0_given_Z0 = observation_model(z0)[Observe_0]
        P_z0 = prior_distribution[z0]
        alpha_z0 = P_O0_given_Z0 * P_z0
        if alpha_z0 > 0:
            alpha_z0_dic[z0] = alpha_z0
    
    forward_messages[0] = rover.Distribution(alpha_z0_dic)
    forward_messages[0].renormalize()

    # recursion
    for i in range(1, num_time_steps):
        alpha_zi_dic = {}
        alpha_zi_1 = forward_messages[i-1]
        Observe_i = observations[i]

        P_Oi_given_Zi = 1
        P_Zi_given_Zi_1 = None
        for zi in all_possible_hidden_states: 
            if Observe_i != None:              
                P_Oi_given_Zi = observation_model(zi)[Observe_i]
            
            sum = 0
            for zi_1 in alpha_zi_1:
                P_Zi_given_Zi_1 = transition_model(zi_1)[zi]
                P_zi = alpha_zi_1[zi_1]*P_Zi_given_Zi_1
                sum += P_zi

            alpha_zi = sum*P_Oi_given_Zi
            if alpha_zi > 0: 
                alpha_zi_dic[zi] = alpha_zi

        forward_messages[i] = rover.Distribution(alpha_zi_dic)
        forward_messages[i].renormalize() 
 
                   
    # ================================ Backward messages ================================
    # initialization 
    #all posible start pos
    beta_z_n_1_dic = {}
    for z_n_1 in all_possible_hidden_states:
        beta_z_n_1_dic[z_n_1] = 1
    backward_messages[num_time_steps-1] = rover.Distribution(beta_z_n_1_dic)

    # recursion
    for i in range(num_time_steps-2,-1,-1):
        beta_zi_dic = {}
        beta_zi_1 = backward_messages[i+1]
        Observe_i1 = observations[i+1]

        P_Oi1_given_Zi1 = 1
        P_Zi1_given_Zi = None

        for zi in all_possible_hidden_states: 
            sum = 0
            for zi1 in beta_zi_1:
                P_Zi1_given_Zi = transition_model(zi)[zi1]
                if Observe_i1 != None:   
                    P_Oi1_given_Zi1 = observation_model(zi1)[Observe_i1]
                P_zi = P_Zi1_given_Zi*P_Oi1_given_Zi1*beta_zi_1[zi1]
                sum += P_zi

            beta_zi = sum
            if beta_zi > 0: 
                beta_zi_dic[zi] = beta_zi

        backward_messages[i] = rover.Distribution(beta_zi_dic)
        backward_messages[i].renormalize() 
   
    
    # ================================ Marginal ================================
    for i in range (0, num_time_steps):   
        marginals_i_dic = {}
        sum = 0
        for zi in all_possible_hidden_states:
            product = forward_messages[i][zi] * backward_messages[i][zi]
            if  product > 0:
                marginals_i_dic[zi] = product
        
        marginals[i] = rover.Distribution(marginals_i_dic)
        marginals[i].renormalize() 

    return marginals
            

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    N = len(observations)
    Delta_List = [None] * N
    Phi_List = [None] * N
    estimated_hidden_states = [None]*N

    # initialization
    Delta_0_dic = {}
    Phi_0_dic = {}
    Observe_0 = observations[0]

    P_O0_given_z0 = 1
    P_z0 = None
    for z0 in all_possible_hidden_states:
        if Observe_0 != None:
            P_O0_given_z0 = observation_model(z0)[Observe_0]
        P_z0 = prior_distribution[z0]
        Delta_0 = P_z0 * P_O0_given_z0
        if Delta_0 > 0:
            Delta_0_dic[z0] = Delta_0
            Phi_0_dic[z0] = 0

    Delta_List[0] = rover.Distribution(Delta_0_dic)
    Delta_List[0].renormalize()
    Phi_List[0] = Phi_0_dic

    # recursion
    for i in range(1, N):
        Phi_i_dic = {}
        Delta_i_dic = {}
        Delta_i_1_dic = Delta_List[i-1]
        Observe_i = observations[i]
        P_Oi_given_zi = 1

        for zi in all_possible_hidden_states:
            if Observe_i != None:
                P_Oi_given_zi = observation_model(zi)[Observe_i]
            
            Max_res = 0
            best_zi_1 = None
            for zi_1 in Delta_i_1_dic:
                Delta_zi_1 = Delta_i_1_dic[zi_1]
                P_Zi_given_Zi_1 = transition_model(zi_1)[zi]
                res_zi_1 = Delta_zi_1 * P_Zi_given_Zi_1
                if(res_zi_1 > Max_res):
                    Max_res = res_zi_1
                    best_zi_1 = zi_1

            Max_res = Max_res*P_Oi_given_zi
            if(Max_res > 0):
                Delta_i_dic[zi] = Max_res
                Phi_i_dic[zi] = best_zi_1

        Delta_List[i] = rover.Distribution(Delta_i_dic)
        Delta_List[i].renormalize()
        Phi_List[i] = Phi_i_dic

    # back track
    Last_Delta_dic = Delta_List[N-1]
    max_delta = 0
    best_zi = None

    for zi in Last_Delta_dic:
        delta_zi = Last_Delta_dic[zi]
        if delta_zi > max_delta:
            max_delta = delta_zi
            best_zi = zi
    estimated_hidden_states[N-1] = best_zi
    
    
    for i in range(N-2,-1,-1):
        Phi1_dic = Phi_List[i+1]
        best_zi1 = estimated_hidden_states[i+1]
        best_zi = Phi1_dic[best_zi1]
        estimated_hidden_states[i] = best_zi

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()     #[(x, y, prev_action) ...]
    all_possible_observed_states = rover.get_all_observed_states()   #[(x_bar, y_bar) ...]
    prior_distribution           = rover.initial_distribution()      #distribution{(x,y,'stay): p}
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


    #Q1 (b) Check 
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])


    # Viterbi_Error
    correct_Viterbi_step = 0
    for i in range(0, num_time_steps):
        if hidden_states[i] == estimated_states[i]:
            correct_Viterbi_step += 1
    Viterbi_Error = 1-correct_Viterbi_step/100
    print("Viterbi_Error:", Viterbi_Error)

    # forward_backward_Error
    correct_forward_backward_step = 0
    forward_backward_step_List = []
    for i in range(0, num_time_steps):
        best_zi = None
        max_p_zi = 0
        Marginal_step_i = marginals[i]
        for zi in Marginal_step_i:
            if Marginal_step_i[zi] > max_p_zi:
                max_p_zi = Marginal_step_i[zi] 
                best_zi = zi
        forward_backward_step_List.append(best_zi)
        if hidden_states[i] == best_zi:
            correct_forward_backward_step += 1
    forward_backward_Error = 1-correct_forward_backward_step/100
    print("forward_backward_Error:", forward_backward_Error)   

    # Check Valid forward_backward
    for i in range(1, num_time_steps):
        prev_state = forward_backward_step_List[i-1]
        this_state = forward_backward_step_List[i]
        prev_x = prev_state[0]
        prev_y = prev_state[1]
        
        this_x = this_state[0]
        this_y = this_state[1]
        prev_dir = this_state[2]

        #['left', 'right', 'up', 'down', 'stay']
        if prev_dir == 'left':
            prev_x -= 1
        if prev_dir == 'right':
            prev_x += 1
        if prev_dir == 'up':
            prev_y -= 1
        if prev_dir == 'down':
            prev_y += 1     

        if(prev_x != this_x or prev_y != this_y):
            print("ERROR:",i-1,":",prev_state)
            print("ERROR:",i,":",this_state)  




  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
