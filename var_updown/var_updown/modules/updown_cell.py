from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from allennlp.nn.util import masked_mean

from updown.modules.attention import BottomUpTopDownAttention


class UpDownCell(nn.Module):

    def __init__(
        self,
        image_feature_size: int,
        embedding_size: int,
        hidden_size: int,
        attention_projection_size: int,
        z_space: int,
        sentiment_vae: int,
        simple_vae,
        device,
        latent_embedding
    ):
        super().__init__()

        self.image_feature_size = image_feature_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_projection_size = attention_projection_size

        self.device = device

        self._attention_lstm_cell = nn.LSTMCell(
            self.embedding_size + self.image_feature_size + 2 * self.hidden_size, self.hidden_size
        )
        self._butd_attention = BottomUpTopDownAttention(
            self.hidden_size, self.image_feature_size, self.attention_projection_size
        )
        
        self.z_space = z_space
        self.sentiment_vae = sentiment_vae
        self.simple_vae = simple_vae
        
        self.latent_embedding = latent_embedding

        if(sentiment_vae == 0):
            self._language_lstm_cell_encoder = nn.LSTMCell(
                self.image_feature_size + 2 * self.hidden_size, self.hidden_size
            )
              
            self._language_lstm_cell_decoder = nn.LSTMCell(
                self.image_feature_size + 2 * self.hidden_size + z_space, self.hidden_size
            )
        elif(self.latent_embedding == "senti_word_net" or sentiment_vae == 1):
            self._language_lstm_cell_encoder = nn.LSTMCell(
                1 + self.image_feature_size + 2 * self.hidden_size, self.hidden_size
            )
              
            self._language_lstm_cell_decoder = nn.LSTMCell(
                1 + self.image_feature_size + 2 * self.hidden_size + z_space, self.hidden_size
            )
        elif(sentiment_vae == 2):
            self._language_lstm_cell_encoder = nn.LSTMCell(
                150 + self.image_feature_size + 2 * self.hidden_size, self.hidden_size
            )
              
            self._language_lstm_cell_decoder = nn.LSTMCell(
                150 + self.image_feature_size + 2 * self.hidden_size + z_space, self.hidden_size
            )
        else:
            raise NotImplementedError()
            
        if(self.simple_vae):
            self._language_lstm_cell_encoder = nn.LSTMCell(
                self.image_feature_size + 2 * self.hidden_size, self.hidden_size
            )
              
            self._language_lstm_cell_decoder = nn.LSTMCell(
                self.image_feature_size + 2 * self.hidden_size + z_space, self.hidden_size
            )            

        self.fc_mean = nn.Linear(self.hidden_size, z_space)             
        self.fc_log_var = nn.Linear(self.hidden_size, z_space)

    def forward(
        self,
        image_features: torch.Tensor,
        obj_atts: None,
        token_embedding: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
        training = True,
        sentiment = None,
        attrib_cond = None,
        prior_mean = None,
        prior_var = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Given image features, input token embeddings of current time-step and LSTM states,
        predict output token embeddings for next time-step and update states. This behaves
        very similar to :class:`~torch.nn.LSTMCell`.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.
        token_embedding: torch.Tensor
            A tensor of shape ``(batch_size, embedding_size)`` containing token embeddings for a
            particular time-step.
        states: Dict[str, torch.Tensor], optional (default = None)
            A dict with keys ``{"h1", "c1", "h2", "c2"}`` of LSTM states: (h1, c1) for Attention
            LSTM and (h2, c2) for Language LSTM. If not provided (at first time-step), these are
            initialized as zeros.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tensor of shape ``(batch_size, hidden_state)`` with output token embedding, which
            is the updated state "h2", and updated states (h1, c1), (h2, c2).
        """
        batch_size = image_features.size(0)
            
        # Average pooling of image features happens only at the first timestep. LRU cache
        # saves compute by not executing the function call in subsequent timesteps.
        # shape: (batch_size, image_feature_size), (batch_size, num_boxes)
        averaged_image_features, image_features_mask = self._average_image_features(image_features)

        # Initialize (h1, c1), (h2, c2) if not passed.
        if states is None:
            state = image_features.new_zeros((batch_size, self.hidden_size))
            states = {
                "h1": state.clone(),
                "c1": state.clone(),
                "h_encoder": state.clone(),
                "c_encoder": state.clone(),
                "h_decoder": state.clone(),
                "c_decoder": state.clone(),
            }

        # shape: (batch_size, embedding_size + image_feature_size + 2 * hidden_size)
        attention_lstm_cell_input = torch.cat(
            [token_embedding, averaged_image_features, states["h1"], states["h_decoder"]], dim=1
        )
        states["h1"], states["c1"] = self._attention_lstm_cell(
            attention_lstm_cell_input, (states["h1"], states["c1"])
        )

        # shape: (batch_size, num_boxes)
        attention_weights = self._butd_attention(
            states["h1"], image_features, image_features_mask=image_features_mask
        )
        
        # shape: (batch_size, image_feature_size)
        attended_image_features = torch.sum(
            attention_weights.unsqueeze(-1) * image_features, dim=1
        )

        if(self.sentiment_vae == 2):
            prior_mean = torch.sum(
                attention_weights.unsqueeze(-1) * obj_atts, dim=1
            )
            
        if(self.simple_vae):
            prior_mean = torch.zeros(prior_mean.shape).to(self.device)
            
    
        if(self.latent_embedding == "glove"):
            c = prior_mean
        elif(self.latent_embedding == "senti_word_net"):
            c = prior_mean[:,0].unsqueeze(1)
        else:
            raise NotImplementedError()
        
        if(training):
            if(self.simple_vae or self.sentiment_vae == 0):
                language_lstm_cell_encoder_input = torch.cat(
                    [attended_image_features, states["h1"], states["h_decoder"]], dim=1
                )
            elif(self.sentiment_vae == 1):
                language_lstm_cell_encoder_input = torch.cat(
                    [attended_image_features, states["h1"], states["h_decoder"], sentiment], dim=1
                )
            elif(self.sentiment_vae == 2):
                language_lstm_cell_encoder_input = torch.cat(
                    [attended_image_features, states["h1"], states["h_decoder"], c], dim=1
                )
            else:
                raise NotImplementedError()

            states["h_encoder"], states["c_encoder"] = self._language_lstm_cell_encoder(
                language_lstm_cell_encoder_input, (states["h_encoder"], states["c_encoder"])
            )
            
            mean = self.fc_mean(states["h_encoder"])
            log_var = self.fc_log_var(states["h_encoder"])
            var = log_var.exp()

        else:
            mean = prior_mean
            var = prior_var
            log_var = var.log()

            
        eps = torch.randn(var.shape).to(self.device)
        
        z = eps * var.sqrt() + mean
        
        
        if(self.simple_vae or self.sentiment_vae == 0):
            language_lstm_cell_decoder_input = torch.cat(
                [attended_image_features, states["h1"], states["h_decoder"], z], dim=1
            )
        elif(self.sentiment_vae == 1):
            language_lstm_cell_decoder_input = torch.cat(
                [attended_image_features, states["h1"], states["h_decoder"], sentiment,  z], dim=1
            )
        elif(self.sentiment_vae == 2):
            language_lstm_cell_decoder_input = torch.cat(
                [attended_image_features, states["h1"], states["h_decoder"], c,  z], dim=1
            )
        else:
            raise NotImplementedError()
        

        states["h_decoder"], states["c_decoder"] = self._language_lstm_cell_decoder(
            language_lstm_cell_decoder_input, (states["h_decoder"], states["c_decoder"])
        )

        return states["h_decoder"], states, mean, log_var, prior_mean, prior_var.log(), attention_weights

    @lru_cache(maxsize=10)
    def _average_image_features(
        self, image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Perform mean pooling of bottom-up image features, while taking care of variable
        ``num_boxes`` in case of adaptive features.

        Extended Summary
        ----------------
        For a single training/evaluation instance, the image features remain the same from first
        time-step to maximum decoding steps. To keep a clean API, we use LRU cache -- which would
        maintain a cache of last 10 return values because on call signature, and not actually
        execute itself if it is called with the same image features seen at least once in last
        10 calls. This saves some computation.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Averaged image features of shape ``(batch_size, image_feature_size)`` and a binary
            mask of shape ``(batch_size, num_boxes)`` which is zero for padded features.
        """
        # shape: (batch_size, num_boxes)
        image_features_mask = torch.sum(torch.abs(image_features), dim=-1) > 0

        # shape: (batch_size, image_feature_size)
        averaged_image_features = masked_mean(
            image_features, image_features_mask.unsqueeze(-1), dim=1
        )

        return averaged_image_features, image_features_mask
