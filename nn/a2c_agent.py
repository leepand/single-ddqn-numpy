from a2c_policy import A2CPolicy
from layers import Dense, Input, Activ
from functional import (
    initialize_parameters,
)  # import function to initialize weights and biases
from model import ModelDB
from optim import Optimizer

# from wrappers import get_activation_layer_function
import copy
import numpy as np
import random


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x))


class A2CAgent:
    def __init__(
        self,
        layers=[(16, "tanh")],
        opt="adam",
        model_db=None,
        lr=0.005,
        eps=0.1,
        gamma=0.99,
        loss="MSE",
    ):
        self.len_actions = 0
        self.actor_net_type = "actor"
        self.critic_net_type = "critic"
        self.loss = loss
        self.stacked = False
        self.opt = opt
        self.optimizer = Optimizer(name=opt, lr=lr)
        self.model_actor = A2CPolicy(
            net_type=self.actor_net_type, optimizer=self.optimizer
        )

        self.model_critic = A2CPolicy(
            net_type=self.critic_net_type, optimizer=self.optimizer
        )
        self.eps = eps
        self.GAMMA = gamma
        self._model_storage = ModelDB(model_db=model_db)
        self.stack_layers(layers=layers)
        self.actions = [a for a in range(self.len_actions)]

    def stack_layers(self, layers: list) -> None:
        """
        Stack the layers of the model.

        Parameters:
        - layers (list): A list containing the layers of the model.
        """
        # Compute the number of layers.
        num_layers = len(layers)
        i = 0
        while i < num_layers:
            # Record the input dimension of the first layer.
            if isinstance(layers[i], Input):
                n_in = layers[i].n_out
                i += 1
                continue
            elif isinstance(layers[i], Dense):
                # ------ LAYER-2/n ----- define output layer that take is values from hidden layer
                n_out = layers[i].n_in
                activation_params = layers[i].activation_params
                activation_name = layers[i].activation_name
                self.model_actor.add(Dense(n_in=n_in, n_out=n_out))
                self.model_critic.add(Dense(n_in=n_in, n_out=n_out))
                self.model_actor.add(
                    Activ(
                        activation=activation_name, activation_params=activation_params
                    )
                )
                self.model_critic.add(
                    Activ(
                        activation=activation_name, activation_params=activation_params
                    )
                )
                self.len_actions = n_out
                n_in = n_out
            i += 1

    def _init_model(self):
        params = {}
        Layerlist_actor = copy.deepcopy(self.model_actor.layers)
        i = 0
        net_type = "actor"
        for L in Layerlist_actor:
            if L.layer_type == "Linear":
                if L.activation_name in ["tanh", "sigmoid"]:
                    int_type = "xavier"
                else:
                    int_type = "he"
                _params = initialize_parameters(
                    L.n_in, L.n_out, ini_type=int_type, opt=self.opt
                )
                params[f"{net_type}_W{i}"] = _params["W"]
                params[f"{net_type}_b{i}"] = _params["b"]
                if self.opt in ["rmsprop", "adam"]:
                    params[f"{net_type}_m1{i}"] = _params["m1"]
                    params[f"{net_type}_m2{i}"] = _params["m2"]
            i += 1
        Layerlist_critic = copy.deepcopy(self.model_critic.layers)
        net_type = "critic"
        j = 0
        for LC in Layerlist_critic:
            if LC.layer_type == "Linear":
                if L.activation_name in ["tanh", "sigmoid"]:
                    int_type = "xavier"
                else:
                    int_type = "he"
                _params = initialize_parameters(
                    LC.n_in, LC.n_out, ini_type=int_type, opt=self.opt
                )
                params[f"{net_type}_W{j}"] = _params["W"]
                params[f"{net_type}_b{j}"] = _params["b"]
                if self.opt in ["rmsprop", "adam"]:
                    params[f"{net_type}_m1{j}"] = _params["m1"]
                    params[f"{net_type}_m2{j}"] = _params["m2"]
            j += 1

        params["model_updated_cnt"] = 0
        return params

    def update_critic_model(self, model):
        j = 0
        Layerlist_critic = copy.deepcopy(self.model_critic.layers)
        for LC in Layerlist_critic:
            if LC.layer_type == "Linear":
                model[f"{self.critic_net_type}_W{j}"] = model[
                    f"{self.actor_net_type}_W{j}"
                ]
                model[f"{self.critic_net_type}_b{j}"] = model[
                    f"{self.actor_net_type}_b{j}"
                ]
            j += 1
        return model

    def act(self, x, model_id, allowed=None, not_allowed=None):
        if isinstance(x, list):
            x = np.array(x).reshape(-1, 1)

        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()

        if allowed is None:
            valid_actions = self.actions
        else:
            valid_actions = allowed
        if not_allowed is not None:
            valid_actions = self._get_valid_actions(forbidden_actions=not_allowed)

        if random.random() < self.eps:

            action = random.choice(valid_actions)
            # print(action,"r")
        else:
            action_probs = self.model_actor.predict(x=x, model=model)
            # print(action_probs, "action_probs")
            action_probs_dict = {a: action_probs[a] for a in valid_actions}
            action = self.argmax_rand(action_probs_dict)
            # print(action,"m",action_probs_dict,action_probs)
        return action

    def learn(
        self,
        state,
        next_state,
        action,
        reward,
        model_id,
        done=False,
        print_cost=False,
    ):
        if isinstance(state, list):
            state = np.array(state).reshape(-1, 1)
        if isinstance(next_state, list):
            next_state = np.array(next_state).reshape(-1, 1)
        if isinstance(reward, (float, int)):
            reward = np.array([reward]).reshape(-1, 1)

        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()

        _action_probs, x_list_actor = self.model_actor.predict(
            x=state, model=model, update_mode=True
        )
        model_updated_cnt = model["model_updated_cnt"]
        if model_updated_cnt % 10 == 0:
            model = self.update_critic_model(model=model)
        # value_curr, x_list_critic = self.model_critic.predict(
        #    x=next_state, model=model, update_mode=True
        # )
        value_next = self.model_critic.predict(next_state, model=model)
        Q = np.copy(_action_probs)
        # print(action,type(action),a,type(a))
        a = np.argmax(value_next, axis=0)

        _action_probs[int(action)] = (
            reward + (1 - np.logical_not(done)) * self.GAMMA * value_next[a]
        )

        model = self.model_actor.fit(
            x_list_actor,
            Q,
            _action_probs,
            model,
            loss_function=self.loss,
            print_cost=print_cost,
        )

        self._model_storage.save_model(model=model, model_id=model_id, w_type="model")

    def max_q_value(self, state, model_id):
        model = self._model_storage.get_model(model_id=model_id, w_type="model")
        if model is None:
            model = self._init_model()
        value_next = self.model_actor.predict(state, model=model)
        a = np.argmax(value_next, axis=0)
        return value_next[a]

    def _get_valid_actions(self, forbidden_actions, all_actions=None):
        """
        Given a set of forbidden action IDs, return a set of valid action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[Set[ActionId]]
            The set of forbidden action IDs.

        Returns
        -------
        valid_actions: Set[ActionId]
            The list of valid (i.e. not forbidden) action IDs.
        """
        if all_actions is None:
            all_actions = self.actions
        if forbidden_actions is None:
            forbidden_actions = set()
        else:
            forbidden_actions = set(forbidden_actions)

        if not all(a in all_actions for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = set(all_actions) - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError(
                "All actions are forbidden. You must allow at least 1 action."
            )

        valid_actions = list(valid_actions)
        return valid_actions

    def argmax_rand(self, dict_arr):
        """Return key with maximum value, break ties randomly."""
        assert isinstance(dict_arr, dict)
        # Find the maximum value in the dictionary
        max_value = max(dict_arr.values())
        # Get a list of keys with the maximum value
        max_keys = [key for key, value in dict_arr.items() if value == max_value]
        # Randomly select one key from the list
        selected_key = random.choice(max_keys)
        # Return the selected key
        return selected_key
