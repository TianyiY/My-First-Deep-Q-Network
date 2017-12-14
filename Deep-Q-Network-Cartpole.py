import numpy as np
import tensorflow as tf
import gym           # simulation environment

np.random.seed(0)     # fix random numbers
tf.set_random_seed(0)

# Set parameters
Learning_rate=0.02
Epsilon=0.88       # probability of applying greed-algorithm
Discount_of_reward=0.9    # gamma in Bellman eqn
Batch_size=64            # how many samples in each learning loop
Storage_samples=4000    # how many samples to store
Update_freq_target=100    # the frequency that replace the target network by the evaluation network

# set simulation environment
Environment=gym.make("CartPole-v0").unwrapped
Actions_Count= Environment.action_space.n    # (move left, move right)
States_Count=Environment.observation_space.shape[0]   # [position x, horizontal velocity x_prime, angle between pole and verticle line theta, angular velocity theta_prime]
print (Actions_Count)
print (States_Count)
print (Environment.x_threshold)
print (Environment.theta_threshold_radians)

# set tensorflow placeholder
State_before=tf.placeholder(tf.float32, [None, States_Count])
Action=tf.placeholder(tf.int32, [None,])
Reward=tf.placeholder(tf.float32, [None,])
State_after=tf.placeholder(tf.float32, [None, States_Count])

# Set target network, update every 200 frequency, no training
with tf.variable_scope('target'):
    Target_=tf.layers.dense(State_after, 16, tf.nn.relu, trainable=False)
    Target=tf.layers.dense(Target_, Actions_Count, trainable=False)

# Set evaluation network for training
with tf.variable_scope('eval'):
    Eval_=tf.layers.dense(State_before, 16, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
    Eval=tf.layers.dense(Eval_, Actions_Count, kernel_initializer=tf.random_normal_initializer(0, 0.1))

# Q-learning for Target and Eval
Q_Target=Reward+Discount_of_reward*tf.reduce_max(Target, axis=1)
Q_Eval=tf.reduce_sum(Eval*tf.one_hot(Action, depth=Actions_Count, dtype=tf.float32), axis=1)

# Training
Loss_score=tf.reduce_mean(tf.squared_difference(Q_Target, Q_Eval))
Training=tf.train.AdamOptimizer(Learning_rate).minimize(Loss_score)

# Initialize tf
sess=tf.Session()
sess.run(tf.global_variables_initializer())

# Function to store parameters
# one sample includes [state_before, action, reward, state_after]
Storage=np.zeros((Storage_samples, States_Count+1+1+States_Count))
def store(S_b, A, R, S_a):
    global Storage_count   # access globally
    Storage_count=0
    Store=np.hstack((S_b, [A, R], S_a))
    i=Storage_count % Storage_samples    # for iteration, reset every storage_samples
    Storage[i,:]=Store   # reset the storage
    Storage_count+=1

# Function to choose best action
def select_action(State):
    State=State[np.newaxis, :]      # add one more dimension
    if np.random.uniform()<Epsilon:
        # apply greedy algorithm
        Action_score=sess.run(Eval, feed_dict={State_before: State})
        Action=np.argmax(Action_score)
    else:
        # explore
        Action=np.random.randint(0, Actions_Count)
    return Action

# Function to learn
def learning():
    # first update target
    global Learning_count
    Learning_count=0
    if Learning_count % Update_freq_target ==0:    # update according to the frequency
        Target_Para=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')   # collect the variables in 'target'
        Eval_Para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')   # collect the variables in 'eval'
        # replace target with eval
        sess.run([tf.assign(T_P, E_P) for T_P, E_P in zip(Target_Para, Eval_Para)])
    Learning_count+=1

    # start learning, iterate per batch size
    i=np.random.choice(Storage_samples, Batch_size)
    Batch_storage=Storage[i, :]
    Batch_State_before=Batch_storage[:, :States_Count]
    Batch_Action=Batch_storage[:, States_Count].astype(int)
    Batch_Reward=Batch_storage[:, States_Count+1]
    Batch_State_after=Batch_storage[:, -States_Count:]
    sess.run(Training, feed_dict={State_before:Batch_State_before,
                                          Action: Batch_Action,
                                          Reward: Batch_Reward,
                                          State_after: Batch_State_after})


print('\nLearning')
Total_step=10000   # 1000 iterations totally
for i in range(Total_step):
    state=Environment.reset()   # reset the environment
    accumulative_Reward=0   # initialize reward = 0
    while True:   # iterate until done
        Environment.render()
        action=select_action(state)

        # make action
        state_after, reward, done, _=Environment.step(action)

        position_x, position_x_prime, angle_theta, angle_theta_prime=state_after   # four elements in state vector

        # you can use default rewards, or define your own reward
        # reward_pos=(Environment.x_threshold-abs(position_x))/Environment.x_threshold-1.0
        # reward_ang=(Environment.theta_threshold_radians-abs(angle_theta))/Environment.theta_threshold_radians-0.5
        # reward=2.0*(0.5*reward_pos+0.5*reward_ang)   # update reward

        store(state, action, reward, state_after)

        accumulative_Reward+=reward

        if Storage_count>Storage_samples:
            learning()
            if done:
                print('Step:', i,
                      'Accumulate Reward:', accumulative_Reward)
        if done:
            break

        state=state_after   # move to the next state and reiterate

'''
Q-learning: does not always work but when it works, usually more sample-efficient. Challenge: exploration
Q-learning: Zero guarantees since you are approximating Bellman equation with a complicated function approximator
'''