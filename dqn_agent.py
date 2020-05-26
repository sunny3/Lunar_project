import gym

# from tensorflow.keras.layers import Dense
# import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
from collections import deque
from sklearn.utils import shuffle
import copy


class QLearningMethod():
    def __init__(self, gamma, learning_rate, batch_size, max_memory_len, n_actions, n_states, eps_settings, tau,
                 mode='train'):
        self.gamma = gamma  # параметр, с помощью которого задается баланс между сиюминутными наградами и будущими
        self.n_actions = n_actions  # количество возможных действий
        self.n_states = n_states  # количество степеней свободы
        self.learning_rate = learning_rate
        self.memory = None  # шаги агента. (состояние_t, действие_t, награда_t, состояние_t+1, выполнен ли эпизод?)
        self.max_memory_len = max_memory_len
        self.batch_size = batch_size
        self.mode = mode
        self.tau = tau
        for key, param in eps_settings.items():
            setattr(self, key, param)
        if self.mode == 'train':
            self.local_model = self.get_QNN()  # сама модель
            self.target_model = self.get_QNN()
        else:
            self.local_model = self.load_QNN()  # обученная модель
        self.n_layers = len(self.local_model.layers)

    def soft_update(self):
        for layer_index in range(self.n_layers):
            local_params = self.local_model.get_layer(index=layer_index).get_weights()
            target_params = self.target_model.get_layer(index=layer_index).get_weights()
            weighted_res = []
            for local_nparray, target_nparray in zip(local_params, target_params):
                weighted_res.append(self.tau * local_nparray + (1 - self.tau) * target_nparray)
            self.target_model.get_layer(index=layer_index).set_weights(weighted_res)

    def get_QNN(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.n_states, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def random_choice(self):
        rand = np.random.random()
        if rand >= 1 - self.eps:
            self.minimize_eps()
            return np.random.choice(range(self.n_actions))
        self.minimize_eps()
        return None

    def minimize_eps(self):
        self.eps = self.eps * self.eps_delta
        if self.eps < self.min_eps and self.n_jumps > 0:
            self.eps = copy.deepcopy(self.max_eps)
            self.n_jumps -= 1
        elif self.eps < self.min_eps:
            self.eps = copy.deepcopy(self.min_eps)

    def load_QNN(self):
        return load_model('./saved_models/big_memory_smart_agent_local.h5')

    def choose_action(self, curr_state):
        if self.mode == 'train':
            choice = self.random_choice()
            if choice is not None:
                return choice
        curr_state = np.reshape(curr_state, [1, self.n_states])
        real_Qvalues = self.local_model.predict(curr_state)
        return np.argmax(real_Qvalues[0])

    def remember_step(self, curr_state, curr_action, curr_reward, next_state, done):
        what2add = np.array([curr_state, curr_action, curr_reward, next_state, done])
        if self.memory is None:
            self.memory = what2add
        else:
            if self.memory.shape[0] == self.max_memory_len:
                self.memory = np.delete(self.memory, 0, axis=0)
            self.memory = np.vstack((self.memory, what2add))

    def faster_expirience_reply(self):
        batch = shuffle(self.memory)[-self.batch_size:]
        states = np.vstack(batch[:, 0])
        real_Qvalues = self.local_model.predict(states)
        non_terminal = batch[:, 4] == False
        non_terminal_indexes = np.array(np.nonzero(non_terminal)).flatten()
        terminal_indexes = np.array(np.nonzero(~non_terminal)).flatten()
        wanted_Qvalues = batch[non_terminal, 2] + self.gamma * np.amax(
            self.target_model.predict(np.vstack(batch[non_terminal, 3])), axis=-1)

        real_Qvalues[non_terminal_indexes, batch[non_terminal_indexes, 1].flatten().astype(np.int64)] = wanted_Qvalues
        real_Qvalues[terminal_indexes, batch[terminal_indexes, 1].flatten().astype(np.int64)] = batch[
            terminal_indexes, 2]
        self.local_model.fit(states, real_Qvalues, batch_size=self.batch_size, epochs=1, verbose=0)
        self.soft_update()

    def expirience_reply(self):
        states = []  # то, что мы должны подать на вход модели: координаты в пространстве
        Qvalues = []  # модель должна выдать полезности всех возможных действий

        #         batch_indexes = np.random.randint(len(self.memory), size = self.batch_size)
        #         batch = [sample for index_sample, sample in enumerate(self.memory) if index_sample in batch_indexes]

        batch = shuffle(self.memory)[-self.batch_size:]
        for curr_state, curr_action, curr_reward, next_state, done in batch:
            states.append(curr_state)
            next_state = np.reshape(next_state, [1, self.n_states])
            curr_state = np.reshape(curr_state, [1, self.n_states])
            wanted_Qvalue = curr_reward + self.gamma * np.amax(self.target_model.predict(next_state)[
                                                                   0]) if not done else curr_reward  # здесь мы предсказываем величину, к которой должна быть приближена полезность действия
            real_Qvalues = self.local_model.predict(curr_state)  # предсказываем полезность всех возможных действий
            real_Qvalues[0][
                curr_action] = wanted_Qvalue  # на место совершенного действия ставим значение для приближения полезности действия
            Qvalues.append(real_Qvalues)
        self.local_model.fit(np.vstack(states), np.vstack(Qvalues), batch_size=self.batch_size, epochs=1, verbose=0)
        self.soft_update()