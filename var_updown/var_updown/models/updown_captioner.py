import functools
import numpy as np
import pickle

import json

import torch
from torch import nn
from torchtext.vocab import GloVe, Vectors
from allennlp.data import Vocabulary
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits

from updown.config import Config
from var_updown.modules import UpDownCell
from updown.modules import ConstrainedBeamSearch
from updown.utils.decoding import select_best_beam, select_best_beam_with_constraints


class UpDownCaptioner(nn.Module):
    def __init__(
        self,
        vocabulary: Vocabulary,
        image_feature_size,
        embedding_size,
        hidden_size,
        attention_projection_size,
        max_caption_length = 20,
        beam_size = 1,
        use_cbs = False,
        min_constraints_to_satisfy = 2,
        z_space = 150,
        prior_std = None,
        simple_vae = False,
        latent_embedding = None,
        latent_embedding_multip = 1,
        sentiment_vae = False,
        senti_prior_multip = 1,
        cbs_simple = False,
        device = None,



    ):
        super().__init__()
        self._vocabulary = vocabulary

        self.image_feature_size = image_feature_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_projection_size = attention_projection_size

        self._max_caption_length = max_caption_length
        self._use_cbs = use_cbs
        self._min_constraints_to_satisfy = min_constraints_to_satisfy

        self.z_space = z_space

        # Short hand variable names for convenience
        _vocab_size = vocabulary.get_vocab_size()
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")
        self._boundary_index = vocabulary.get_token_index("@@BOUNDARY@@")

        self.prior_std = prior_std
        self.sentiment_vae = sentiment_vae
        self.senti_prior_multip = senti_prior_multip
        self.simple_vae = simple_vae
        
        self.latent_embedding = latent_embedding
        self.latent_embedding_multip = latent_embedding_multip

        # Initialize embedding layer with GloVe embeddings and freeze it if the specified size
        # is 300. CBS cannot be supported for any other embedding size, using CBS is optional
        # with embedding size 300. So in either cases, embeddig size is the deciding factor.
        if self.embedding_size == 300 or self.embedding_size == 600:
            if(sentiment_vae == 2):
                glove_vectors = self._initialize_glove()
       
                self.senti_glove_10 = pickle.load(open("/path/to/sentiglove10.pkl", "rb" ))
                for k,v in self.senti_glove_10.items():
                    self.senti_glove_10[k] = np.repeat(v, (self.z_space / 10))
                
                with open("/path/to/wordform_swd_scores.json", "r") as read_file:
                   self.senti_wordnet_scores = json.load(read_file)                
                for k,v in self.senti_wordnet_scores.items():
                    self.senti_wordnet_scores[k] = np.repeat(v[0]-v[2], (self.z_space))
                
                if(latent_embedding == "glove"):
                    self.mean_choice = self.senti_glove_5
                elif(latent_embedding == "senti_word_net"):
                    self.mean_choice = self.senti_wordnet_scores
                else:
                    raise NotImplementedError()
                
            else:
                glove_vectors = self._initialize_glove()
                
            self._embedding_layer = nn.Embedding.from_pretrained(
                glove_vectors, freeze=True, padding_idx=self._pad_index
            )
        else:
            self._embedding_layer = nn.Embedding(
                _vocab_size, embedding_size, padding_idx=self._pad_index
            )
            assert not use_cbs, "CBS is not supported without Frozen GloVe embeddings (300d), "
            f"found embedding size to be {self.embedding_size}."

        self._updown_cell = UpDownCell(
            image_feature_size, embedding_size, hidden_size, attention_projection_size, z_space, sentiment_vae, simple_vae, device, latent_embedding
        )

        if self.embedding_size == 300 or self.embedding_size == 600:
            # Tie the input and output word embeddings when using frozen GloVe embeddings.
            # In this case, project hidden states to GloVe dimension (with a non-linearity).
            self._output_projection = nn.Sequential(
                nn.Linear(hidden_size, self.embedding_size), nn.Tanh()
            )
            self._output_layer = nn.Linear(self.embedding_size, _vocab_size, bias=False)
            self._output_layer.weight = self._embedding_layer.weight
        else:
            # Else don't tie them when learning embeddings during training.
            # In this case, project hidden states directly to output vocab space.
            self._output_projection = nn.Identity()  # type: ignore
            self._output_layer = nn.Linear(hidden_size, _vocab_size)

        self._log_softmax = nn.LogSoftmax(dim=1)

        # We use beam search to find the most likely caption during inference.
        BeamSearchClass = ConstrainedBeamSearch if use_cbs else BeamSearch
        self._beam_search = BeamSearchClass(
            self._boundary_index,
            max_steps=max_caption_length,
            beam_size=beam_size,
            per_node_beam_size=beam_size // 2,
        )
        
        self.device = device
        self.cbs_simple = cbs_simple
        
        
    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config

        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            image_feature_size=_C.MODEL.IMAGE_FEATURE_SIZE,
            embedding_size=_C.MODEL.EMBEDDING_SIZE,
            hidden_size=_C.MODEL.HIDDEN_SIZE,
            attention_projection_size=_C.MODEL.ATTENTION_PROJECTION_SIZE,
            beam_size=_C.MODEL.BEAM_SIZE,
            max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
            use_cbs=_C.MODEL.USE_CBS,
            min_constraints_to_satisfy=_C.MODEL.MIN_CONSTRAINTS_TO_SATISFY,
            z_space = _C.MODEL.Z_SPACE,
            prior_std = _C.MODEL.PRIOR_STD,
            simple_vae = _C.MODEL.SIMPLE_VAE,
            latent_embedding = _C.MODEL.LATENT_EMBEDDING,
            sentiment_vae = _C.MODEL.SENTIMENT_VAE,
            senti_prior_multip = _C.MODEL.SENTI_PRIOR_MULTIP,
            latent_embedding_multip = _C.MODEL.LATENT_EMBEDDING_MULTIP,
            cbs_simple=_C.MODEL.CBS_SIMPLE,
            device=kwargs["device"],
        )

    def _initialize_glove(self):
        r"""
        Initialize embeddings of all the tokens in a given
        :class:`~allennlp.data.vocabulary.Vocabulary` by their GloVe vectors.

        Extended Summary
        ----------------
        It is recommended to train an :class:`~updown.models.updown_captioner.UpDownCaptioner` with
        frozen word embeddings when one wishes to perform Constrained Beam Search decoding during
        inference. This is because the constraint words may not appear in caption vocabulary (out of
        domain), and their embeddings will never be updated during training. Initializing with frozen
        GloVe embeddings is helpful, because they capture more meaningful semantics than randomly
        initialized embeddings.

        Returns
        -------
        torch.Tensor
            GloVe Embeddings corresponding to tokens.
        """
        
        if(self.embedding_size == 300):        
            glove = GloVe(name="42B", dim=300, cache="/path/to/.vector_cache")
            glove_vectors = torch.zeros(self._vocabulary.get_vocab_size(), 300)

            for word, i in self._vocabulary.get_token_to_index_vocabulary().items():
                if word in glove.stoi:
                    glove_vectors[i] = glove.vectors[glove.stoi[word]]
                elif word != self._pad_index:
                    # Initialize by random vector.
                    glove_vectors[i] = 2 * torch.randn(300) - 1
        elif(self.embedding_size == 600):
            glove = GloVe(name="42B", dim=300, cache="/path/to/.vector_cache")
            dependency_embedding = Vectors(name="deps.words", cache="/path/to/.vector_cache")

            glove_vectors = torch.zeros(self._vocabulary.get_vocab_size(), 600)
    
            for word, i in self._vocabulary.get_token_to_index_vocabulary().items():
                if word in glove.stoi:
                    v1 = glove.vectors[glove.stoi[word]]
                elif word != self._pad_index:
                    # Initialize by random vector.
                    v1 = 2 * torch.randn(300) - 1
                    
                if word in dependency_embedding.stoi:
                    v2 = dependency_embedding.vectors[dependency_embedding.stoi[word]]
                elif word != self._pad_index:
                    # Initialize by random vector.
                    v2 = 2 * torch.randn(300) - 1
                    
                glove_vectors[i] = torch.cat((v1, v2), 0)
                   
        else:
            raise NotImplementedError()
                        
        return glove_vectors
            

                    
        return glove_vectors 

    def forward(  # type: ignore
        self,
        image_features: torch.Tensor,
        obj_atts = None,
        image_attributes = None,
        caption_tokens = None,
        sentiment = None,
        fsm: torch.Tensor = None,
        num_constraints: torch.Tensor = None,
        constraints = None,
        constraint2states = None,
    ):
        #print("image_features.size()", image_features.size())
        batch_size, num_boxes, image_feature_size = image_features.size()

        # Initialize states at zero-th timestep.
        states = None

        if(self.sentiment_vae == 2 and obj_atts is not None):
            obj_atts = self.translate_obj_atts2obj_means(obj_atts)
        
        attrib_cond = None
        if self.sentiment_vae == 0:
            prior_mean = torch.zeros((batch_size, self.z_space)).to(self.device)
        elif self.sentiment_vae == 1:
            prior_mean = sentiment.repeat(1, self.z_space) * self.senti_prior_multip
        elif self.sentiment_vae == 2:
            prior_mean = torch.zeros((batch_size, self.z_space)).to(self.device)
            attrib_cond = None
        else:
            raise NotImplementedError()
            
        prior_var = (torch.ones(prior_mean.shape).to(self.device) * self.prior_std).pow(2)
        prior_log_var = prior_var.log()

        if self.training and caption_tokens is not None:
            # Add "@@BOUNDARY@@" tokens to caption sequences.
            caption_tokens, _ = add_sentence_boundary_token_ids(
                caption_tokens,
                (caption_tokens != self._pad_index),
                self._boundary_index,
                self._boundary_index,
            )
            batch_size, max_caption_length = caption_tokens.size()

            # shape: (batch_size, max_caption_length)
            tokens_mask = caption_tokens != self._pad_index

            # The last input from the target is either padding or the boundary token.
            # Either way, we don't have to process it.
            num_decoding_steps = max_caption_length - 1

            step_logits = []
            step_klds = []
            for timestep in range(num_decoding_steps):
                # shape: (batch_size,)
                input_tokens = caption_tokens[:, timestep]

                # shape: (batch_size, num_classes)
                output_logits, states, q_mean, q_log_var, prior_mean, prior_log_var, attention_weights = self._decode_step(image_features,
                                                                                                obj_atts,
                                                                                                input_tokens, 
                                                                                                states, 
                                                                                                sentiment, 
                                                                                                attrib_cond, 
                                                                                                prior_mean, 
                                                                                                prior_var)
                q_var = q_log_var.exp()


                if self.sentiment_vae == 0:
                    kld = -0.5 * torch.sum(1 + q_log_var - q_mean.pow(2) - q_log_var.exp(), dim = 1)
                else: 
                    kld = 1 + q_log_var - prior_log_var - \
                        ((q_mean - prior_mean).pow(2) + q_var) / (prior_var + 0.00001)
                    kld = -0.5 * kld.sum(1)


                    
                # list of tensors, shape: (batch_size, 1, vocab_size)
                step_logits.append(output_logits.unsqueeze(1))
                step_klds.append(kld.unsqueeze(1))
                
            # shape: (batch_size, num_decoding_steps)
            logits = torch.cat(step_logits, 1)
            
            
            klds = torch.cat(step_klds, 1) * tokens_mask[:, 1:].float()
            
            # Skip first time-step from targets for calculating loss.
            output_dict = {
                "loss": self._get_loss(
                    logits, caption_tokens[:, 1:].contiguous(), tokens_mask[:, 1:].contiguous()
                ),
                "kld": torch.sum(klds, dim=1)
            }
        else:
            num_decoding_steps = self._max_caption_length
            start_predictions = image_features.new_full((batch_size,), self._boundary_index).long()

            # Add image features as a default argument to match callable signature acceptable by
            # beam search class (previous predictions and states only).
            beam_decode_step = functools.partial(self._decode_step, 
                                                 image_features,
                                                 obj_atts,
                                                 sentiment=sentiment, 
                                                 attrib_cond=attrib_cond,
                                                 prior_mean=prior_mean, 
                                                 prior_var=prior_var)

            # shape (all_top_k_predictions): (batch_size, net_beam_size, num_decoding_steps)
            # shape (log_probabilities): (batch_size, net_beam_size)
            
            output_dict = dict()
            if self._use_cbs:
                all_top_k_predictions, log_probabilities = self._beam_search.search(
                    start_predictions, states, beam_decode_step, fsm
                )

                best_beam, valid_beams = select_best_beam_with_constraints(
                    all_top_k_predictions,
                    log_probabilities,
                    num_constraints,
                    constraints,
                    constraint2states,
                    self._min_constraints_to_satisfy,
                    self.cbs_simple,
                )
                
                output_dict["valid_beams"] = valid_beams
                
            else:
                all_top_k_predictions, log_probabilities = self._beam_search.search(
                    start_predictions, states, beam_decode_step
                )
                best_beam = select_best_beam(all_top_k_predictions, log_probabilities)

            # shape: (batch_size, num_decoding_steps)
            output_dict = {"predictions": best_beam}

        return output_dict


    def _decode_step(
        self,
        image_features,
        obj_atts,
        previous_predictions,
        states = None,
        sentiment = None,
        attrib_cond = None,
        prior_mean = None,
        prior_var = None,
    ):
        r"""
        Given image features, tokens predicted at previous time-step and LSTM states of the
        :class:`~updown.modules.updown_cell.UpDownCell`, take a decoding step. This is also
        called by the beam search class.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``.
        previous_predictions: torch.Tensor
            A tensor of shape ``(batch_size * net_beam_size, )`` containing tokens predicted at
            previous time-step -- one for each beam, for each instances in a batch.
            ``net_beam_size`` is 1 during teacher forcing (training), ``beam_size`` for regular
            :class:`allennlp.nn.beam_search.BeamSearch` and ``beam_size * num_states`` for
            :class:`updown.modules.cbs.ConstrainedBeamSearch`

        states: [Dict[str, torch.Tensor], optional (default = None)
            LSTM states of the :class:`~updown.modules.updown_cell.UpDownCell`. These are
            initialized as zero tensors if not provided (at first time-step).
        """
        net_beam_size = 1

        # Expand and repeat image features while doing beam search (during inference).
        if not self.training and image_features.size(0) != previous_predictions.size(0):

            batch_size, num_boxes, image_feature_size = image_features.size()
            net_beam_size = int(previous_predictions.size(0) / batch_size)

            # Add (net) beam dimension and repeat image features.
            image_features = image_features.unsqueeze(1).repeat(1, net_beam_size, 1, 1)
            
            # shape: (batch_size * net_beam_size, num_boxes, image_feature_size)
            image_features = image_features.view(
                batch_size * net_beam_size, num_boxes, image_feature_size
            )
            
            if(self.sentiment_vae == 1):
                sentiment = sentiment.repeat(net_beam_size, 1)
            else:
                pass
                
            prior_mean = prior_mean.repeat(net_beam_size, 1)
            prior_var = prior_var.repeat(net_beam_size, 1)

        # shape: (batch_size * net_beam_size, )
        current_input = previous_predictions

        # shape: (batch_size * net_beam_size, embedding_size)
        token_embeddings = self._embedding_layer(current_input)

        # shape: (batch_size * net_beam_size, hidden_size)
        updown_output, states, mean, log_var, prior_mean, prior_log_var, attention_weights = self._updown_cell(image_features, 
                                                                 obj_atts,
                                                                 token_embeddings, 
                                                                 states, 
                                                                 self.training, 
                                                                 sentiment,
                                                                 attrib_cond,
                                                                 prior_mean,
                                                                 prior_var)

        # shape: (batch_size * net_beam_size, vocab_size)
        updown_output = self._output_projection(updown_output)
        output_logits = self._output_layer(updown_output)

        # Return logits while training, to further calculate cross entropy loss.
        # Return logprobs during inference, because beam search needs them.
        # Note:: This means NO BEAM SEARCH DURING TRAINING.
        outputs = output_logits if self.training else self._log_softmax(output_logits)

        if self.training:
            return outputs, states, mean, log_var, prior_mean, prior_log_var, attention_weights  # type: ignore
        else:
            return outputs, states, prior_mean, prior_log_var, attention_weights

    def _get_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
    ):
        # shape: (batch_size, )
        target_lengths = torch.sum(target_mask, dim=-1).float()

        # shape: (batch_size, )
        return target_lengths * sequence_cross_entropy_with_logits(
            logits, targets, target_mask, average=None
        )
    
    def batch_calc_attrib_mean(self, batch_image_attribs):
        
        batch_result_150 = []
        batch_result_10 = []

        
        for item in batch_image_attribs:
            weight_sum = 0
            item_mean_150 = torch.zeros(self.z_space) 
            item_mean_10 = torch.zeros(10) 
            
            att_words = []
            att_weights = []
            if item:
                for o in item:
                    for a in o[1]:
                        a_cleaned = a[0].split(" ")[-1]
                        if not a_cleaned:
                            a_cleaned = a[0].split(" ")[-2]
                        if a_cleaned not in att_words:
                            att_words.append(a_cleaned)
                            att_weights.append(a[1])
                        else:
                            a_idx = att_words.index(a_cleaned)
                            att_weights[a_idx] = max(att_weights[a_idx], a[1])

                for a_word, a_weight in zip(att_words, att_weights):
                    item_mean_150 += self.senti_glove_150[a_word] * a_weight
                    item_mean_10 += self.senti_glove_10[a_word] * a_weight
                weight_sum += a_weight
            
            if(weight_sum > 0):
               item_mean_150 /= weight_sum
               item_mean_10 /= weight_sum

            batch_result_150.append(item_mean_150)
            batch_result_10.append(item_mean_10)
            
        
        return torch.stack(batch_result_150).to(self.device), torch.stack(batch_result_10).to(self.device)
    
    def translate_obj_atts2obj_means(self, obj_atts):
        result = []
        for im in obj_atts:
            means = []
            for obj in im:
                mean = []
                for att in obj[1]:
                    try:
                        mean.append(self.mean_choice[att.split(" ")[0]])
                    except:
                        pass
                if len(mean):
                    means.append(np.mean(mean, axis=0))
                else:
                    means.append(np.zeros(self.z_space))
            result.append(np.array(means))
        
        lens = [len(x) for x in result]
        result_new = np.zeros((len(result), max(lens), self.z_space))
        for i_i, im in enumerate(result):
            for i_o, obj in enumerate(im):
                result_new[i_i, i_o, :] = obj

        return torch.Tensor(result_new * self.latent_embedding_multip).to(self.device)