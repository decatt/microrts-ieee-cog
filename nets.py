import torch
import torch.nn as nn
from utils import layer_init, MaskedCategorical

class ActorCriticMix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*2*2, 128)),
            nn.ReLU(),
        )

        self.encoder = nn.GRU(29, 128, batch_first=True)
        self.decoder = nn.GRU(128, 128, batch_first=True)

        self.dis = layer_init(nn.Linear(256, 78), std=0.01)
        
        self.value = layer_init(nn.Linear(256, 1), std=1)
        
    def forward(self,cnn_states,linears_states):
        cnn_states = cnn_states.permute((0, 3, 1, 2))
        z_cnn = self.encoder_cnn(cnn_states)

        batch_size = linears_states.size(0)
        seq_len = linears_states.size(1)

        # Encoder
        encoder_outputs, hidden = self.encoder(linears_states)

        # Decoder
        decoder_state = hidden
        decoder_input = torch.zeros((batch_size, 1, 128)).to(linears_states.device)
        decoder_outputs = []

        for t in range(seq_len):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        z_pn = decoder_outputs[-1][:,-1,:]

        policy_network = torch.cat((z_cnn,z_pn),dim=1)

        distris = self.dis(policy_network)
        
        value = self.value(policy_network)

        return distris, value

class ActorCritic(nn.Module):
    def __init__(self,cnn_output_dim,action_space) -> None:
        super().__init__()
        self.policy_network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(cnn_output_dim, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, sum(action_space)))
        
        self.value = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(cnn_output_dim, 256)),
                nn.ReLU(), 
                layer_init(nn.Linear(256, 1), std=1)
            )
        
    def get_distris(self,states):
        states = states.permute((0, 3, 1, 2))
        policy_network = self.policy_network(states)
        action_dist = self.actor(policy_network)
        return action_dist

    def get_value(self, states):
        states = states.permute((0, 3, 1, 2))
        value = self.value(states)
        return value
    
    def forward(self, states):
        distris = self.get_distris(states)
        value = self.get_value(states)
        return distris,value