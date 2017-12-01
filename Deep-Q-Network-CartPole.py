import numpy as np
import tensorflow as tf
import gym           # simulation environment

np.random.seed(12)
tf.set_random_seed(12)

# Set parameters
Learning_rate=0.02
Epsilon=0.95       # probability of applying greed-algorithm
Discount_of_reward=0.9
Batch_size=64
Storage_samples=5000    # how many samples to store
Update_freq_target=200

# set simulation environment
Environment=gym.make("CartPole-v0").unwrapped
Actions_Count= Environment.action_space.n
States_Count=Environment.observation_space.shape[0]

# set tensorflow placeholder
State_before=tf.placeholder(tf.float64, [None, States_Count])
Action=tf.placeholder(tf.int64, [None,])
Reward=tf.placeholder(tf.float64, [None,])
State_after=tf.placeholder(tf.float64, [None, States_Count])

# Set target network, update every 200 frequency, no training
with tf.variable_scope('target'):
    Target_=tf.layers.dense(State_after, 5, tf.nn.relu, trainable=False)
    Target=tf.layers.dense(Target_, Actions_Count, trainable=False)

# Set evaluation network for training
with tf.variable_scope('eval'):
    Eval_=tf.layers.dense(State_before, 5, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.01))
    Eval=tf.layers.dense(Eval_, Actions_Count, kernel_initializer=tf.random_normal_initializer(0, 0.01))

# Q-learning for Target and Eval
Q_Target=Reward+Discount_of_reward*tf.reduce_max(Target, axis=1)
Q_Eval=tf.reduce_sum(Eval*tf.one_hot(Action, depth=Actions_Count, dtype=tf.float64), axis=1)

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
    i=Storage_count % Storage_samples    # for iteration
    Storage[i,:]=Store   # reset the storage
    Storage_count+=1

# Function to choose best action
def select_action(State):
    State=State[np.newaxis, :]
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
    if Learning_count % Update_freq_target ==0:
        Target_Para=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        Eval_Para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        # replace target with eval
        sess.run([tf.assign(T_P, E_P) for T_P, E_P in zip(Target_Para, Eval_Para)])
    Learning_count+=1

    # start learning
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
Total_step=1000
for i in range(Total_step):
    state=Environment.reset()
    accumulative_Reward=0
    while True:
        Environment.render()
        action=select_action(state)

        # make action
        state_after, reward, done, _=Environment.step(action)

        position, position_, angle, angle_=state_after
        reward_pos=(Environment.x_threshold-abs(position))/Environment.x_threshold
        reward_ang=(Environment.theta_threshold_radians-abs(angle))/Environment.theta_threshold_radians
        reward=reward_pos+reward_ang   # update reward

        store(state, action, reward, state_after)

        accumulative_Reward+=reward

        if Storage_count>Storage_samples:
            learning()
            if done:
                print('Step:', i,
                      'Accumulate Reward:', accumulative_Reward)
        if done:
            break

        state=state_after


