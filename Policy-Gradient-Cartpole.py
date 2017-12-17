import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

np.random.seed(0)    # fix random numbers
tf.set_random_seed(0)

Threshold_Render = 500  # the threshold (reward) that render the environment
RENDER = False  # do not render default

Environment = gym.make('CartPole-v0')    # CartPole environment
Environment.seed(0)     # you'd better fix the random numbers since general Policy gradient has high variance
Environment = Environment.unwrapped

print(Environment.action_space)
print(Environment.observation_space)
# print(Environment.observation_space.high)
# print(Environment.observation_space.low)


class PolicyGradient:

    def __init__(self, Actions_Count, States_Count, Learning_rate=0.02, Discount_of_reward=0.90, graph_output=False):
        self.Actions_Count = Actions_Count     # number of actions (move left, move right)
        self.States_Count = States_Count     # (horizontal position, horizontal velocity, angle, angular velocity)
        self.Learning_rate = Learning_rate
        self.Discount_of_reward = Discount_of_reward   # gamma, reward discount
        self.States, self.Actions, self.Rewards = [], [], []    # initialize storage
        self.initialize_Policy_Gradient_network()    # initialize network
        self.sess = tf.Session()

        if graph_output:    # output tensowflow graph
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # output the graph
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def initialize_Policy_Gradient_network(self):
        with tf.name_scope('tf_variables'):
            self.tf_states = tf.placeholder(tf.float32, [None, self.States_Count], name="tf_states")
            self.tf_actions = tf.placeholder(tf.int32, [None, ], name="tf_actions")
            self.tf_qvalues = tf.placeholder(tf.float32, [None, ], name="tf_qvalues")

        # fully connected layer 1 (4->16)
        layer1 = tf.layers.dense(
            inputs=self.tf_states,
            units=16,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fully_connected_layer_1'
        )
        # fully connected layer 2 (16->2)
        layer2 = tf.layers.dense(
            inputs=layer1,
            units=self.Actions_Count,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fully_connected_layer_2'
        )

        self.actions_prob = tf.nn.softmax(layer2, name='actions_prob')  # use softmax to convert to probability (0-1)

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), since the tf only have minimize(loss)
            negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=self.tf_actions)   # this is negative log of chosen action
            # or in this way:
            # negative_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(negative_log_prob * self.tf_qvalues)  # reward guided loss

        with tf.name_scope('train'):
            self.training = tf.train.AdamOptimizer(self.Learning_rate).minimize(loss)

    def select_action(self, state):
        prob_weights = self.sess.run(self.actions_prob, feed_dict={self.tf_states: state[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store(self, state, action, reward):
        self.States.append(state)
        self.Actions.append(action)
        self.Rewards.append(reward)

    def rewards_discounted(self):
        # discounted rewards
        Reward_Discounted = np.zeros_like(self.Rewards)
        summ = 0
        for i in reversed(range(0, len(self.Rewards))):
            summ = summ * self.Discount_of_reward + self.Rewards[i]
            Reward_Discounted[i] = summ

        # normalize rewards
        Reward_Discounted -= np.mean(Reward_Discounted)
        Reward_Discounted /= np.std(Reward_Discounted)
        return Reward_Discounted

    def learning(self):
        # discount and normalize episode reward
        Reward_DN = self.rewards_discounted()

        # train on episode
        self.sess.run(self.training, feed_dict={
             self.tf_states: np.vstack(self.States),  # shape=[None, n_obs]
             self.tf_actions: np.array(self.Actions),  # shape=[None, ]
             self.tf_qvalues: Reward_DN,  # shape=[None, ]
        })

        self.States, self.Actions, self.Rewards = [], [], []    # empty episode data
        return Reward_DN


RL = PolicyGradient(
    Actions_Count=Environment.action_space.n,
    States_Count=Environment.observation_space.shape[0],
    Learning_rate=0.02,
    Discount_of_reward=0.90,
    # output_graph=True,
)

for i_episode in range(5000):

    state = Environment.reset()

    while True:

        if RENDER:
            Environment.render()

        action = RL.select_action(state)

        state_, reward, done, info = Environment.step(action)

        RL.store(state, action, reward)

        if done:
            Rewards_sum = sum(RL.Rewards)

            if 'Reward_Running' not in globals():
                Reward_Running = Rewards_sum
            else:
                Reward_Running = Reward_Running * 0.99 + Rewards_sum * 0.01
            if Reward_Running > Threshold_Render:
                RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(Reward_Running))

            value = RL.learning()

            if i_episode == 0:
                plt.plot(value)    # plot the episode value vs time
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        state = state_


'''
 Policy gradients: very general but suffer from high variance so requires a lot of samples. Challenge: sample-efficiency
 Policy Gradients: Converges to a local minima of J(), often good enough! 
'''