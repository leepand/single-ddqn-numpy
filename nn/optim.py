import numpy as np


class Optimizer(object):
    """
    An optimizer selector.
    """

    def __init__(self, name: str, lr=0.001, eta=3e-4) -> None:
        """
        Initialize the optimizer selector.

        Parameters:
        - name (str): The name of the optimizer.
        """
        self.name = name
        if name == "sgd":
            self.lr = lr
        elif name in ["rmsprop", "adam"]:
            self.eta = eta  # 1e-3
            self.b1 = 0.9  # Adam, if b1 = 0. -> Adam = RMSprop
            self.b2 = 0.99
            self.eps = 1e-10

    def update_params(self, dW, db, W, b, m1=None, m2=None, model_updated_cnt=None):
        """
        This function performs the gradient descent update

        Args:
            learning_rate: learning rate hyper-param for gradient descent, default 0.1
            adam's lr: self.eta
        """
        if self.name in ["adam", "rmsprop"]:  # RMSprop
            # print(m1["W"].shape, dW.shape)
            m1["W"] = self.b1 * m1["W"] + (1 - self.b1) * dW
            m1["b"] = self.b1 * m1["b"] + (1 - self.b1) * db
            m2["W"] = self.b2 * m2["W"] + (1 - self.b2) * dW**2
            m2["b"] = self.b2 * m2["b"] + (1 - self.b2) * db**2
            if self.name == "rmsprop":
                W += self.eta * dW / (np.sqrt(m2["W"]) + self.eps)
                b += self.eta * db / (np.sqrt(m2["b"]) + self.eps)
            elif self.name == "adam":
                m_b_hat = m1["b"] / (
                    1.0 - self.b1 ** (model_updated_cnt + 1)
                )  # bias correction
                v_b_hat = m2["b"] / (
                    1.0 - self.b2 ** (model_updated_cnt + 1)
                )  # bias correction
                m_w_hat = m1["W"] / (
                    1.0 - self.b1 ** (model_updated_cnt + 1)
                )  # bias correction
                v_w_hat = m2["W"] / (
                    1.0 - self.b2 ** (model_updated_cnt + 1)
                )  # bias correction
                # print(v_b_hat,m2["b"],"v_b_hat")
                W -= self.eta * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
                b -= self.eta * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
                # W += self.eta * m1["W"] / (np.sqrt(m2["W"]) + self.eps)
                # W += self.eta * m1["W"] / (np.sqrt(m2["W"]) + self.eps)
                # print(b,m1["b"] ,m2["b"],"llll")
                # b += self.eta * m1["b"] / (np.sqrt(m2["b"]) + self.eps)
            else:
                raise ValueError(f"{self.name} is not supported.")
        else:  # normal
            W = W - self.lr * dW  # update weights
            b = b - self.lr * db  # update bias(es)
        if self.name in ["adam", "rmsprop"]:
            return W, b, m1, m2
        else:
            return W, b
