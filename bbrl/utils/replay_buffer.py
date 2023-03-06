# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import torch

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
        self.best_reward = 1


    def init_workspace(self, all_tensors):
        """
        Create an array to stores workspace based on the given all_tensors keys.
        shape of stores tensors : [key] => [self.max_size][time_size][key_dim]
        Makes a copy of the input content
        """
        #if the variables are not already set, we set them
        if self.variables is None:
            self.variables = {}
            for k, v in all_tensors.items():
                #in order to solve the problem of the end of an episode and the beginning of another,
                #we replace single step informations with transition informations
                s = list(v.size())
                s[1] = self.max_size
                _s = copy.deepcopy(s)
                s[0] = _s[1]
                s[1] = _s[0]

                tensor = torch.zeros(*s, dtype=v.dtype, device=self.device)
                self.variables[k] = tensor
            self.is_full = False
            self.position = 0


    #function which insert the value v into the variable dictionnary of the replay buffer
    def _insert(self, k, indexes, v):
        self.variables[k][indexes] = v.detach().moveaxis((0, 1), (1, 0))


    def put(self, workspace):
        """
        Add a the content of a workspace to the replay buffer.
        The given workspace must have keys of shape : [time_size][batch_size][key_dim]
        """
        #recovery of position from the workspace
        new_data = {
            k: workspace.get_full(k).detach().to(self.device) for k in workspace.keys()
        }
        self.init_workspace(new_data)

        batch_size = None
        arange = None
        indexes = None

        for k, v in new_data.items():
            if batch_size is None:
                batch_size = v.size()[1]  #if the batch_size is not set, we set it equal to the size of the variable of the
                #workspace that we are adding in the Replay Buffer (new_data corresponds to the dictionnary of the
                #workspace related to variables, see the file workspace.py, line 330 in the __init__)

                # print(f"{k}: batch size : {batch_size}")
                # print("pos", self.position)

            #we check if we can add a batch of the size of batch_size in the replay buffer
            if self.position + batch_size < self.max_size:
                #the case where the batch can be inserted before the end of the replay buffer
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


    #function which gives the size of the replay buffer
    def size(self):
        if self.is_full:
            return self.max_size
        else:
            return self.position


    #function which print the observations
    def print_obs(self):
        print(f"position: {self.position}")
        print(self.variables["env/env_obs"])


    #A MODIFIER POUR NE PLUS PRENDRE LES BATCH RANDOMLY MAIS EN FONCTION DE L'EMPLACEMENT DU BATCH DANS LE REPLAY BUFFER
    #LES BATCHS LES PLUS A GAUCHE (CEUX SONT LES PLUS PERTINENTS POUR L'APPRENTISSAGE) ONT UNE PROBA ELEVEE D'ÊTRE TIRES
    #AU LIEU DE PRENDRE LES BATCH RANDOMLY, ON LES TIRE SELON UNE DISTRIBUTION TIREE VERS LA GAUCHE
    #LOI DE POISSON AVEC 0<alpha<=1

    #function which allows to get the batchs picked according to a Poisson distribution from the replay buffer (under the form of workspace)
    def get_shuffled(self, batch_size):
        rates = torch.rand(size=1)*batch_size  #rate parameter between 0 and 1 (of the size of the batch_size)
        who = torch.poisson(rates) #torch.poisson alows to pick a tensor with a probability distribution shifted
                                   #in order to have a higher probability for a batch which is more relevant for the learning
        workspace = Workspace() #the tensor 'who' contains thus the INDEXS of the values of the replay buffer, picked according to the Poisson distribution
        #for k in self.variables:
            #workspace.set_full(k, self.variables[k][].transpose(0, 1))  #a ajuster !

        return workspace


    def get_prioritized(self, batch_size):
        workspace=Workspace()
        rewards = []
        samples_priorities = []
        batch = []

        for k in self.variables:
            reward = variable[k]['reward']  #quel emplacement ?
            rewards.append(reward)  #on récupère les rewards de chaque sample
        self.best_reward = max(rewards)  #on récupère le plus grand reward

        for k in self.variables:
            reward = variable[k]["reward"]
            priority = reward/self.best_reward  #pour chaque sample on définit la priorité en fonction du rapport entre
                                                #son reward et le meilleur reward -> 0<priorities<1
            samples_priorities.append({'sample': k, 'priority': priority})  #on associe à chaque sample sa priorité et on
                                                                            #regroupe tous les couples dans un dictionnaire
                    #liste de dictionnaire et non de tuples car si besoin on peut changer la valeur de la priorité

        samples_priorities.sort(key=['priority'])  #on trie la liste des dictionnaires en fonction des priorités
        #afin d'effectuer par la suite un tirage avec une proba plus élevée sur les samples avec une plus grande priorité



        #IDEE 1:
        for i in range(batch_size):  #on créé un batch selon la taille désirée en entrée
            batch.append(samples_priorities[i])

        return batch  #on renvoit le batch

        #IDEE 2:
        workspace = Workspace()
        #for k in samples_priorities:
         #   workspace.set_full(k, samples_priorities[k][].transpose(0, 1))  #a ajuster !

        return workspace




    def to(self, device):
        n_vars = {k: v.to(device) for k, v in self.variables.items()}
        self.variables = n_vars
