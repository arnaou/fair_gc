##########################################################################################################
#                                                                                                        #
#    Script for fitting training graph neural networks                                                   #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#                                                                                                        #
#    Authors: Adem R.N. Aouichaoui                                                                       #
#    2024/12/03                                                                                          #
#                                                                                                        #
##########################################################################################################

import torch

class AttentiveFPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, edge_dim=None, num_layers=3, num_timesteps=3, num_mlp=2):
        super().__init__()