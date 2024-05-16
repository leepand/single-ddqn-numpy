from abc import ABCMeta, abstractmethod
import numpy as np
import copy


class Policy(metaclass=ABCMeta):
    def __init__(self, input_n, output_n, hidden_n=64, hidden_layer_n=2, lr=1e-4):
        pass

    @abstractmethod
    def train(self, x, y, learning_rate=0.001):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def act(self, x):
        pass

    @abstractmethod
    def learn(self, x, action, next_x, reward, done):
        pass

    @classmethod
    def ReLU(cls, x):
        return np.maximum(0, x)

    @classmethod
    def d_ReLU(cls, x):
        return np.heaviside(x, 1.0)

    @classmethod
    def Sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def d_Sigmoid(cls, x):
        return (1 - cls.Sigmoid(x)) * cls.Sigmoid(x)

    @classmethod
    def tanh(self, x):
        return np.tanh(x)

    @classmethod
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2


class MLPPolicy(Policy):
    # For weight initialization, used He normal init.
    # For bias initialization, used Zero init.
    def __init__(self, input_n, output_n, hidden_n=16, hidden_layer_n=1, eps=0.05):
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.hidden_layer_n = hidden_layer_n
        self.layers = self._init_model()
        self.target_Q_layers = self._init_model()
        self.eps = eps
        self.action_n = output_n

    def _init_model(self):
        layers = list()
        input_w = np.random.normal(
            scale=np.sqrt(2 / self.input_n), size=(self.input_n, self.hidden_n)
        )
        input_b = np.zeros(self.hidden_n)

        layers.append((input_w, input_b))

        for i in range(self.hidden_layer_n):
            hidden_w = np.random.normal(
                scale=np.sqrt(2 / self.hidden_n), size=(self.hidden_n, self.hidden_n)
            )
            hidden_b = np.zeros(self.hidden_n)

            layers.append((hidden_w, hidden_b))

        output_w = np.random.normal(
            scale=np.sqrt(2 / self.hidden_n), size=(self.hidden_n, self.output_n)
        )
        output_b = np.zeros(self.output_n)
        layers.append((output_w, output_b))
        return layers

    def train(self, x, y, learning_rate=1e-4):  # 0.00003):
        predict, update_helper = self.predict(x, update_mode=True)
        update_layers = list()
        d = predict - y
        cost = np.mean(np.square(d))
        d = d * 2
        reversed_layers = list(reversed(self.layers))

        for i, info in enumerate(update_helper):
            # C is cost, x is node
            prev_layer, mid_layer, layer, activate_d_func = info
            # dC/df'(wx+b)
            dl = d
            if activate_d_func is None:
                dl_a = dl
            else:
                # Calculate f'(wx + b) * dC/df'(wx+b)
                dl_a = activate_d_func(mid_layer) * dl

            # dC/db = 1 * f'(wx + b) * dC/df'(wx+b)
            db = np.mean(dl_a, axis=0)
            # dC/dw = x * f'(wx + b) * dC/df'(wx+b)
            # y.shape[0] for mean of total gradient
            dw = prev_layer.T @ dl_a / y.shape[0]

            w, b = reversed_layers[i]

            # For backpropagation, we need tensor shape of (batchsize, output)
            # which has each node's gradient for each batch.
            # dC/dx = df(wx+b)/dx * dC/df(wx+b) = w * f'(wx + b) * dC/df'(wx+b) = w * dl_a
            d = (w @ dl_a.T).T
            update_layers.append((dw, db))

        update_layers.reverse()

        for i, l in enumerate(zip(update_layers, self.layers)):
            update_layer, layer = l
            dw, db = update_layer
            w, b = layer
            # use for clipping gradient
            # dw = np.clip(dw, -0.5, 0.5)
            # db = np.clip(db, -0.5, 0.5)
            w -= learning_rate * dw
            b -= learning_rate * db
            self.layers[i] = (w, b)

        return cost

    def act(self, x):
        if self.eps <= np.random.random():
            action = np.random.randint(0, self.action_n)
        else:
            flat_state = np.reshape(x, (-1))[True, :]
            Q_value = self.predict([flat_state])
            action = np.argmax(Q_value)

        return action

    def learn(
        self, x, action, next_x, reward, done, learning_rate=1e-4, update_Q=False
    ):
        # target_Q_layers = np.copy(self.layers)
        # print(target_Q_layers,"target_Q_layers0")
        state = np.reshape(x, (-1))
        states = state[True, :]
        predict, update_helper = self.predict(states, update_mode=True)

        update_layers = list()
        if update_Q:
            self.target_Q_layers = copy.deepcopy(self.layers)
        next_states = np.reshape(next_x, (-1))[True, :]
        # print(next_states,"next_states")
        target_Q_value = self.predict(next_states, layers=self.target_Q_layers)
        # print(target_Q_value,"target_Q_value")
        Q_value = np.copy(predict)
        if done:
            Q_value[0, action] = reward
        else:
            Q_value[0, action] = reward + 0.99 * np.max(target_Q_value[0])
        y = Q_value
        d = predict - Q_value
        cost = np.mean(np.square(d))
        d = d * 2
        reversed_layers = list(reversed(self.layers))

        for i, info in enumerate(update_helper):
            # C is cost, x is node
            prev_layer, mid_layer, layer, activate_d_func = info
            # dC/df'(wx+b)
            dl = d
            if activate_d_func is None:
                dl_a = dl
            else:
                # Calculate f'(wx + b) * dC/df'(wx+b)
                dl_a = activate_d_func(mid_layer) * dl

            # dC/db = 1 * f'(wx + b) * dC/df'(wx+b)
            db = np.mean(dl_a, axis=0)
            # dC/dw = x * f'(wx + b) * dC/df'(wx+b)
            # y.shape[0] for mean of total gradient
            dw = prev_layer.T @ dl_a / y.shape[0]

            w, b = reversed_layers[i]

            # For backpropagation, we need tensor shape of (batchsize, output)
            # which has each node's gradient for each batch.
            # dC/dx = df(wx+b)/dx * dC/df(wx+b) = w * f'(wx + b) * dC/df'(wx+b) = w * dl_a
            d = (w @ dl_a.T).T
            update_layers.append((dw, db))

        update_layers.reverse()

        for i, l in enumerate(zip(update_layers, self.layers)):
            update_layer, layer = l
            dw, db = update_layer
            w, b = layer
            # use for clipping gradient
            # dw = np.clip(dw, -0.5, 0.5)
            # db = np.clip(db, -0.5, 0.5)
            w -= learning_rate * dw
            b -= learning_rate * db
            self.layers[i] = (w, b)

        return cost

    def predict(self, x, update_mode=False, layers=None):
        update_helper = list()
        if layers is None:
            layers = self.layers
        prev_x = x
        for param in layers[:-1]:
            w, b = param
            mid_x = x @ w + b
            x = self.tanh(mid_x)
            if update_mode:
                update_helper.append((prev_x, mid_x, x, self.tanh_derivative))
                prev_x = np.copy(x)

        w, b = layers[-1]
        if update_mode:
            mid_x = x @ w + b
            update_helper.append((x, mid_x, mid_x, None))
            return x @ w + b, list(reversed(update_helper))
        else:
            return x @ w + b


if __name__ == "__main__":
    p = MLPPolicy(2, 1, hidden_n=16, hidden_layer_n=2)
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[100], [0], [0], [1]])

    for i in range(1000):

        cost = p.train(x, y)
        if i % 100 == 0:
            print(cost)

    print(p.predict([[0, 0]]))
