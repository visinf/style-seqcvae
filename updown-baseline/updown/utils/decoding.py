import csv
from typing import Dict, List

import numpy as np

import torch
from allennlp.data import Vocabulary


def select_best_beam(
    beams: torch.Tensor,
    beam_log_probabilities: torch.Tensor,
) -> torch.Tensor:
    r"""
    Select the best beam out of a set of decoded beams.

    Parameters
    ----------
    beams: torch.Tensor
        A tensor of shape ``(batch_size, num_states, max_decoding_steps)`` containing decoded
        beams by :class:`~allennlp.nn.beam_search.BeamSearch`. These beams are sorted according to
        their likelihood (descending) in ``beam_size`` dimension.
    beam_log_probabilities: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size)`` containing likelihood of decoded
        beams.
    """
    return beams[:, 0, :]


def select_best_beam_with_constraints(
    beams: torch.Tensor,
    beam_log_probabilities: torch.Tensor,
    given_constraints: torch.Tensor,
    constraints,
    constraint2states,
    min_constraints_to_satisfy: int = 2,
    cbs_simple=True,
) -> torch.Tensor:
    r"""
    Select the best beam which satisfies specified minimum constraints out of a total number of
    given constraints.

    .. note::

        The implementation of this function goes hand-in-hand with the FSM building implementation
        in :meth:`~updown.utils.constraints.FiniteStateMachineBuilder.build` - it defines which
        state satisfies which (basically, how many) constraints. If the "definition" of states
        change, then selection of beams also changes accordingly.

    Parameters
    ----------
    beams: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size, max_decoding_steps)`` containing
        decoded beams by :class:`~updown.modules.cbs.ConstrainedBeamSearch`. These beams are
        sorted according to their likelihood (descending) in ``beam_size`` dimension.
    beam_log_probabilities: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size)`` containing likelihood of decoded
        beams.
    given_constraints: torch.Tensor
        A tensor of shape ``(batch_size, )`` containing number of constraints given at the start
        of decoding.
    min_constraints_to_satisfy: int, optional (default = 2)
        Minimum number of constraints to satisfy. This is either 2, or ``given_constraints`` if
        they are less than 2. Beams corresponding to states not satisfying at least these number
        of constraints will be dropped. Only up to 3 supported.

    Returns
    -------
    torch.Tensor
        Decoded sequence (beam) which has highest likelihood among beams satisfying constraints.
    """
    batch_size, num_states, beam_size, max_decoding_steps = beams.size()

    best_beams: List[torch.Tensor] = []
    best_beam_log_probabilities: List[torch.Tensor] = []
    
    batch_valid_beams = []
    
    for i in range(batch_size):
        # fmt: off
        
        if(cbs_simple):
            valid_states = [
                s for s in range(2 ** given_constraints[i].item())
                if bin(s).count("1") >= min(given_constraints[i], min_constraints_to_satisfy)
            ]
        else:            
            valid_object_states = []
    
            states_per_constraint = []
            
            states_objects = np.zeros(2 ** given_constraints[i].item(), dtype=int)
    
            objects_with_attributes = np.zeros(2 ** given_constraints[i].item(), dtype=int) 
            
            for o in constraints[i]:
                states_object = np.zeros(2 ** given_constraints[i].item(), dtype=int)
                states_idx = constraint2states[i][o[0]]
                states_object[states_idx] = 1
                
                states_attributes = np.zeros(2 ** given_constraints[i].item(), dtype=int)
                
                if(not o[1]):
                    states_attributes[:] = 1
                else:
                    for a in o[1]:
                        states_attribute = np.zeros(2 ** given_constraints[i].item(), dtype=int)                    
                        states_idx = constraint2states[i][a]
                        states_attribute[states_idx] = 1
                        states_attributes |= states_attribute
                
                states_object &= states_attributes
                
                if(not np.all(states_attributes)):
                    objects_with_attributes |= states_object
                
                states_objects += states_object
            #print(states_objects)
            
            if(np.any(objects_with_attributes)):
                states_objects *= (np.clip(states_objects, 0, 1) & objects_with_attributes)
            #print(states_objects)
            valid_states = np.where(states_objects >= min(len(constraints[i]), min_constraints_to_satisfy))[0]

        #raise SystemExit()
        # fmt: on

        valid_beams = beams[i, valid_states, 0, :]
        batch_valid_beams.append(valid_beams)
        valid_beam_log_probabilities = beam_log_probabilities[i, valid_states, 0]

        selected_index = torch.argmax(valid_beam_log_probabilities)
        best_beams.append(valid_beams[selected_index, :])
        best_beam_log_probabilities.append(valid_beam_log_probabilities[selected_index])


    # shape: (batch_size, max_decoding_steps)
    return torch.stack(best_beams).long().to(beams.device), torch.stack(batch_valid_beams).to(beams.device)
