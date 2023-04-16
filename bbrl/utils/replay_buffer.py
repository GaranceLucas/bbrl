# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import torch
import numpy as np

from bbrl.workspace import Workspace


class ReplayBuffer:
    def __init__(self, max_size, device=torch.device("cpu")):
        self.max_size = int(max_size)
        self.variables = None
        self.position = 0
        self.is_full = False
        self.device = device

    def init_workspace(self, all_tensors):
        """
        Create an array to stores workspace based on the given all_tensors keys.
        shape of stores tensors : [key] => [self.max_size][time_size][key_dim]
        Makes a copy of the input content
        """

        if self.variables is None:
            self.variables = {}
            for k, v in all_tensors.items():
                s = list(v.size())
                s[1] = self.max_size
                _s = copy.deepcopy(s)
                s[0] = _s[1]
                s[1] = _s[0]

                tensor = torch.zeros(*s, dtype=v.dtype, device=self.device)
                self.variables[k] = tensor
            self.is_full = False
            self.position = 0

    def _insert(self, k, indexes, v):
        self.variables[k][indexes] = v.detach().moveaxis((0, 1), (1, 0))

    def put(self, workspace):
        """
        Add a the content of a workspace to the replay buffer.
        The given workspace must have keys of shape : [time_size][batch_size][key_dim]
        """

        new_data = {
            k: workspace.get_full(k).detach().to(self.device) for k in workspace.keys()
        }
        self.init_workspace(new_data)

        batch_size = None
        arange = None
        indexes = None

        for k, v in new_data.items():
            if batch_size is None:
                batch_size = v.size()[1]
                # print(f"{k}: batch size : {batch_size}")
                # print("pos", self.position)
            if self.position + batch_size < self.max_size:
                # The case where the batch can be inserted before the end of the replay buffer
                if indexes is None:
                    indexes = torch.arange(batch_size) + self.position
                    arange = torch.arange(batch_size)
                    self.position = self.position + batch_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                # print("insertion standard:", indexes)
                # # print("v shape", v.detach().shape)
                self._insert(k, indexes, v)
            else:
                # The case where the batch cannot be inserted before the end of the replay buffer
                # A part is at the end, the other part is in the beginning
                self.is_full = True
                # the number of data at the end of the RB
                batch_end_size = self.max_size - self.position
                # the number of data at the beginning of the RB
                batch_begin_size = batch_size - batch_end_size
                if indexes is None:
                    # print(f"{k}: batch size : {batch_size}")
                    # print("pos", self.position)
                    # the part of the indexes at the end of the RB
                    indexes = torch.arange(batch_end_size) + self.position
                    arange = torch.arange(batch_end_size)
                    # the part of the indexes at the beginning of the RB
                    # print("insertion intermediate computed:", indexes)
                    indexes = torch.cat((indexes, torch.arange(batch_begin_size)), 0)
                    arange = torch.cat((arange, torch.arange(batch_begin_size)), 0)
                    # print("insertion full:", indexes)
                    self.position = batch_begin_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                self._insert(k, indexes, v)

    def size(self):
        if self.is_full:
            return self.max_size
        else:
            return self.position

    def print_obs(self):
        print(f"position: {self.position}")
        print(self.variables["env/env_obs"])

    def get_shuffled(self, batch_size):
        who = torch.randint(
            low=0, high=self.size(), size=(batch_size,), device=self.device
        )
        workspace = Workspace()
        for k in self.variables:
            workspace.set_full(k, self.variables[k][who].transpose(0, 1))

        return workspace

    def to(self, device):
        n_vars = {k: v.to(device) for k, v in self.variables.items()}
        self.variables = n_vars


class PrioritizedReplayBuffer:
    def __init__(self, max_size, device=torch.device("cpu")):
        self.max_size = int(max_size)
        self.variables = None
        self.position = 0
        self.is_full = False
        self.device = device
        self.gamma = 0.9
        self.lambda_exp = 0.2

    def init_workspace(self, all_tensors):
        """
        Create an array to stores workspace based on the given all_tensors keys.
        shape of stores tensors : [key] => [self.max_size][time_size][key_dim]
        Makes a copy of the input content
        """
        # if the variables are not already set, we set them
        if self.variables is None:
            self.variables = {}
            for k, v in all_tensors.items():
                # in order to solve the problem of the end of an episode and the beginning of another,
                # we replace single step information with transition information
                s = list(v.size())
                s[1] = self.max_size
                _s = copy.deepcopy(s)
                s[0] = _s[1]
                s[1] = _s[0]

                tensor = torch.zeros(*s, dtype=v.dtype, device=self.device)
                self.variables[k] = tensor
            self.is_full = False
            self.position = 0

    def _insert(self, k, indexes, v):
        """
        Inserts the value v into the variable dictionnary of the replay buffer
        """
        self.variables[k][indexes] = v.detach().moveaxis((0, 1), (1, 0))

    def put(self, workspace):
        """
        Add the content of a workspace to the replay buffer.
        The given workspace must have keys of shape : [time_size][batch_size][key_dim]
        """
        # recovery of position from the workspace
        new_data = {
            k: workspace.get_full(k).detach().to(self.device) for k in workspace.keys()
        }
        self.init_workspace(new_data)

        batch_size = None
        arange = None
        indexes = None

        for k, v in new_data.items():
            if batch_size is None:
                batch_size = v.size()[
                    1]  # if the batch_size is not set, we set it equal to the size of the variable of the
                # workspace that we are adding in the Replay Buffer (new_data corresponds to the dictionnary of the
                # workspace related to variables, see the file workspace.py, line 330 in the __init__)

                # print(f"{k}: batch size : {batch_size}")
                # print("pos", self.position)

            # we check if we can add a batch of the size of batch_size in the replay buffer
            if self.position + batch_size < self.max_size:
                # the case where the batch can be inserted before the end of the replay buffer
                if indexes is None:
                    indexes = torch.arange(batch_size) + self.position
                    arange = torch.arange(batch_size)
                    self.position = self.position + batch_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                # print("insertion standard:", indexes)
                # # print("v shape", v.detach().shape)
                self._insert(k, indexes, v)

            else:
                # The case where the batch cannot be inserted before the end of the replay buffer
                # A part is at the end, the other part is in the beginning
                self.is_full = True
                # the number of data at the end of the RB
                batch_end_size = self.max_size - self.position
                # the number of data at the beginning of the RB
                batch_begin_size = batch_size - batch_end_size
                if indexes is None:
                    # print(f"{k}: batch size : {batch_size}")
                    # print("pos", self.position)
                    # the part of the indexes at the end of the RB
                    indexes = torch.arange(batch_end_size) + self.position
                    arange = torch.arange(batch_end_size)
                    # the part of the indexes at the beginning of the RB
                    # print("insertion intermediate computed:", indexes)
                    indexes = torch.cat((indexes, torch.arange(batch_begin_size)), 0)
                    arange = torch.cat((arange, torch.arange(batch_begin_size)), 0)
                    # print("insertion full:", indexes)
                    self.position = batch_begin_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                self._insert(k, indexes, v)

    def size(self):
        """
        Returns the size of the replay buffer
        """
        if self.is_full:
            return self.max_size
        else:
            return self.position

    def print_obs(self):
        """
        Print the observation of the replay buffer at the current position
        """
        print(f"position: {self.position}")
        print(self.variables["env/env_obs"])

    def get_shuffled(self, batch_size):
        """
        Returns a workspace with a batch of data randomly picked from the replay buffer
        """
        who = torch.randint(
            low=0, high=self.size(), size=(batch_size,), device=self.device
        )
        workspace = Workspace()
        for k in self.variables:
            workspace.set_full(k, self.variables[k][who].transpose(0, 1))

        return workspace

    def get_prioritized(self, batch_size):
        """
        Returns a workspace with a batch of data picked from the replay buffer according to an exponential distribution
        """
        temporal_differences = []
        batch = [0] * batch_size

        # compute the temporal differences for all the samples
        for k in range(len(self.variables['env/reward'])):
            reward = self.variables['env/reward'][k][0]
            next_reward = self.variables['env/reward'][k][1]
            q_value = self.variables['q_values'][k][0]
            q_value_next = self.variables['q_values'][k][1]
            action = self.variables['action'][k]
            temporal_differences.append(
                reward + self.gamma * max(q_value_next) - q_value[action[0]])

        #get the sorted list according to the temporal differences and the indexes
        tds, indexes = torch.sort(torch.tensor(temporal_differences))

        # create the batch to write into the workspace
        # indexes are picked from the exponential distribution (shifted in order to pick with a higher probability the indexes with the highest temporal differences)
        for i in range(batch_size):
            chosen_index = int((-1 / self.lambda_exp) * np.log(
                np.random.exponential(scale=self.lambda_exp, size=None) * (1 / self.lambda_exp)))
            # the index has to be between 0 and batch_size
            while chosen_index > batch_size or chosen_index < 0:
                chosen_index = int((-1 / self.lambda_exp) * np.log(
                    np.random.exponential(scale=self.lambda_exp, size=None) * (1 / self.lambda_exp)))
            batch[i] = tds[chosen_index]

        who = torch.tensor(batch)
        who = who.long()

        workspace = Workspace()
        for k in self.variables:
            workspace.set_full(k, self.variables[k][who].transpose(0, 1))

        return workspace

    def to(self, device):
        n_vars = {k: v.to(device) for k, v in self.variables.items()}
        self.variables = n_vars



class ReplayBufferCounter:
    def __init__(self, max_size_buffer, max_counter = 1, max_size_used_samples = 200, device=torch.device("cpu")):
        self.max_size_buffer = int(max_size_buffer)
        self.variables = None
        self.position = 0
        self.is_full = False
        self.device = device
        self.max_counter = max_counter
        self.counter_list = [0]*max_size_buffer
        self.used_samples = []
        self.max_size_used_samples = max_size_used_samples

    def init_workspace(self, all_tensors):
        """
        Create an array to stores workspace based on the given all_tensors keys.
        shape of stores tensors : [key] => [self.max_size_buffer][time_size][key_dim]
        Makes a copy of the input content
        """

        if self.variables is None:
            self.variables = {}
            for k, v in all_tensors.items():
                s = list(v.size())
                s[1] = self.max_size_buffer
                _s = copy.deepcopy(s)
                s[0] = _s[1]
                s[1] = _s[0]

                tensor = torch.zeros(*s, dtype=v.dtype, device=self.device)
                self.variables[k] = tensor
            self.is_full = False
            self.position = 0

    def _insert(self, k, indexes, v):
        """
        Inserts the value v into the variable dictionnary of the replay buffer
        """
        self.variables[k][indexes] = v.detach().moveaxis((0, 1), (1, 0))

    def update_counter(self, index):
        """
        Update the counter of the sample at the given index in the replay buffer.
        If the counter reaches the max_counter, the sample is deleted from the replay buffer.
        """
        self.counter_list[index] += 1
        # if the counter of the sample is equals to the max_counter, we add the sample to the list of
        # used samples
        if self.counter_list[index] == self.max_counter:
            for k in range(len(self.variables)):
                self.used_samples.append(self.variables[k][index])
            # if the number of used samples reach the maximum number, we create another replay buffer
            # which does not contain the used samples
            if len(self.used_samples) == self.max_size_used_samples:
                self.new_buffer()

    def new_buffer(self):
        """
        Create another replay buffer from the old replay buffer, without containing the used samples.
        """
        samples_list = []
        # samples_list contains the samples which does not have their counter equals to max_counter
        for k in range(len(self.variables)):
            for j in range(len(self.variables[k])):
                if self.counter_list[self.variables[k][j]] != self.max_counter:
                    samples_list.append(self.variables[k][j])
        # creation of the new replay buffer
        new_buffer_size = self.max_size_buffer - self.max_size_used_samples
        ReplayBufferCounter(new_buffer_size, self.max_counter, self.max_size_used_samples, self.device)
        # insertion of the samples in the new replay buffer
        for s in samples_list:
            self.put(s)

    def put(self, workspace):
        """
        Add the content of a workspace to the replay buffer.
        The given workspace must have keys of shape : [time_size][batch_size][key_dim]
        """
        new_data = {
            k: workspace.get_full(k).detach().to(self.device) for k in workspace.keys()
        }
        self.init_workspace(new_data)

        batch_size = None
        arange = None
        indexes = None

        for k, v in new_data.items():
            if batch_size is None:
                batch_size = v.size()[1]
                # print(f"{k}: batch size : {batch_size}")
                # print("pos", self.position)
            if self.position + batch_size < self.max_size_buffer:
                # The case where the batch can be inserted before the end of the replay buffer
                if indexes is None:
                    indexes = torch.arange(batch_size) + self.position
                    arange = torch.arange(batch_size)
                    self.position = self.position + batch_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                # print("insertion standard:", indexes)
                # # print("v shape", v.detach().shape)
                self._insert(k, indexes, v)
            else:
                # The case where the batch cannot be inserted before the end of the replay buffer
                # A part is at the end, the other part is in the beginning
                self.is_full = True
                # the number of data at the end of the RB
                batch_end_size = self.max_size_buffer - self.position
                # the number of data at the beginning of the RB
                batch_begin_size = batch_size - batch_end_size
                if indexes is None:
                    # print(f"{k}: batch size : {batch_size}")
                    # print("pos", self.position)
                    # the part of the indexes at the end of the RB
                    indexes = torch.arange(batch_end_size) + self.position
                    arange = torch.arange(batch_end_size)
                    # the part of the indexes at the beginning of the RB
                    # print("insertion intermediate computed:", indexes)
                    indexes = torch.cat((indexes, torch.arange(batch_begin_size)), 0)
                    arange = torch.cat((arange, torch.arange(batch_begin_size)), 0)
                    # print("insertion full:", indexes)
                    self.position = batch_begin_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                self._insert(k, indexes, v)

    def size(self):
        """
        Returns the size of the replay buffer
        """
        if self.is_full:
            return self.max_size_buffer
        else:
            return self.position

    def print_obs(self):
        """
        Print the observation of the replay buffer at the current position
        """
        print(f"position: {self.position}")
        print(self.variables["env/env_obs"])

    def get_shuffled(self, batch_size):
        """
        Returns a workspace with a batch of data randomly picked from the replay buffer.
        Takes care to do not pick a sample which have its counter equals to the max_counter.
        """
        who = torch.randint(
            low=0, high=self.size(), size=(batch_size,), device=self.device
        )

        # while we pick a sample which have its counter equals to the max_counter, we pick again another sample
        while self.counter_list[who] == self.max_counter:
            who = torch.randint(
                low=0, high=self.size(), size=(batch_size,), device=self.device
            )

        workspace = Workspace()
        for k in self.variables:
            workspace.set_full(k, self.variables[k][who].transpose(0, 1))
        self.update_counter(who)
        return workspace

    def to(self, device):
        n_vars = {k: v.to(device) for k, v in self.variables.items()}
        self.variables = n_vars

