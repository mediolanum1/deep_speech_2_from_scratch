import re
from string import ascii_lowercase
from collections import defaultdict
import torch


# TODO add BPE, LM support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""
    EMPTY_IND = 0

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()


    def ctc_decode(self, inds) -> str:
          """
          Decoding with CTC.
          Used to decode output of the model

          Args: 
            inds (list): list of tokens.
          Returs: 
            raw_text (str): raw text without empty tokens nor repetitions.
          """
          decoded = []
          last_char_ind = self.EMPTY_IND
          for ind in inds:
             if last_char_ind == ind:
                continue
             if ind != self.EMPTY_IND:
                decoded.append(self.ind2char[ind])
                last_char_ind = ind

          return "".join(decoded)

# This function expands and merges paths based on the next character probabilities.
# params: dp - dict storing possible paths(prefixes) and their probs up to current timestep
#         next_token_probs - probs of next possible chars

# basically creates dict and stores all possible paths, for each next_token_prob compares with our last char to avoid reps
# in the end multiplies current prob v with next_token_prob that we choose
    def expand_and_merge_path(self,dp, next_token_probs):
      new_dp = defaultdict(float)
      for ind, next_token_prob in enumerate(next_token_probs):
        current_char = self.ind2char[ind]
        for (prefix, last_char), v in dp.items():
          if last_char == current_char:
            new_prefix = prefix
          else:
            if current_char != self.EMPTY_TOK:
              new_prefix = prefix + current_char
            else:
              new_prefix = prefix
          new_dp[(new_prefix, current_char)] += v * next_token_prob
      return new_dp
    
    # This function keeps only the top beam_size paths with the highest probabilities.
    def truncate_paths(self, dp, beam_size):
      return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])
    
    
    def ctc_beam_search(self, probs, beam_size):
      dp = {
          ('', self.EMPTY_TOK): 1.0,  # dp is initialized with a single path ('', EMPTY_TOK) and probability 1.0.
      }
      for prob in probs:
        dp = self.expand_and_merge_path(dp, prob)
        dp = self.truncate_paths(dp,beam_size)
      dp = [
        (prefix, proba / len(prefix) if len(prefix) > 0 else proba)
        for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
      ]
      return dp

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

