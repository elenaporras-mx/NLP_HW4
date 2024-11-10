#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###

        WB = 0.01*torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

        # Verify indices match expectations
        assert self.tagset.index(Tag('C')) == 0
        assert self.tagset.index(Tag('H')) == 1
        assert self.eos_t == 2
        assert self.bos_t == 3
        assert self.vocab.index(Word('1')) == 0
        assert self.vocab.index(Word('2')) == 1
        assert self.vocab.index(Word('3')) == 2

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # Debug: Print initial counts
        print("\nBefore M-step:")
        print("A counts:\n", self.A_counts)
        print("B counts:\n", self.B_counts)

        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
        assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'


        # given : Update emission probabilities (self.B).
        self.B_counts += λ          # smooth the counts (EOS_WORD and BOS_WORD remain at 0 since they're not in the matrix)
        self.B = self.B_counts / self.B_counts.sum(dim=1, keepdim=True)  # normalize into prob distributions
        self.B[self.eos_t, :] = 0   # RESOLVED - replace these nan values with structural zeroes, just as in init_params
        self.B[self.bos_t, :] = 0

     
        if self.unigram: #uni case 
            row_counts = self.A_counts.sum(dim=0) + λ  # sum over previous tags
            WA = torch.log(row_counts).unsqueeze(0)  # make it a 1xk matrix
            WA[:, self.bos_t] = -float('inf')
             # normalize same as init 
            self.A = WA.softmax(dim=1) 
            self.A = self.A.repeat(self.k, 1) # copy step
        else:
            # bigram model
            WA = torch.log(self.A_counts + λ)
            WA[:, self.bos_t] = -float('inf') 
            WA[self.eos_t, :] = -float('inf') 
            # normalize same as init
            self.A = WA.softmax(dim=1) 

        # Debug: Print final matrices
        print("\nAfter M-step:")
        print("A matrix:\n", self.A)
        print("B matrix:\n", self.B)
        
        # Debug: Print row sums to verify normalization
        print("\nVerification:")
        print("A row sums:", self.A.sum(dim=1))
        print("B row sums:", self.B.sum(dim=1))
        
        # Print specific probabilities that match spreadsheet
        print("\nKey probabilities (compare with spreadsheet):")
        print(f"p(C|H) = {self.A[1,0]:.4f}")  # H->C transition
        print(f"p(H|C) = {self.A[0,1]:.4f}")  # C->H transition
        print(f"p(1|C) = {self.B[0,0]:.4f}")  # C emits 1
        print(f"p(3|H) = {self.B[1,2]:.4f}")  # H emits 3

    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path] = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      
      
        dev_loss = loss(self)   # evaluate the model at the start of training
        
        old_dev_loss: float = dev_loss     # loss from the last epoch
        step: int = 0   # total number of sentences the model has been trained on so far      
        while step < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # For convenience when working in a Python notebook, 
        # we automatically save our training work by default.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        
        print("\nProcessing sentence:", isent)

        # currently stuck here .. the indexes are weird and other strange errors
        for j in range(1, len(isent)):
            word_id, tag_id = isent[j]
            prev_word_id, prev_tag_id = isent[j-1] # I think this is good to have, but it could also b useless
            
            if tag_id is not None:  # we have a tag
                
                self.B_counts[tag_id, word_id] += mult
                print(f"Added emission count: tag={tag_id}, word={word_id}, count={mult}")
                
                if prev_tag_id is not None:
                    self.A_counts[prev_tag_id, tag_id] += mult
                    print(f"Added transition count: prev_tag={prev_tag_id}, tag={tag_id}, count={mult}")
        
        # Debug output: show accumulated counts
        print("\nCurrent counts after this sentence:")
        print("A_counts:\n", self.A_counts)
        print("B_counts:\n", self.B_counts)
        
    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the 
        forward probability will marginalize over all possible tags.  
        
        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""
        
        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list alpha of length n+2 so that we can assign 
        # directly to each alpha[j] in turn.
        
        # same as the first thing - the viterbi 
        n = len(isent) - 2  # exclude BOS and EOS
        valid_tags = [t for t in range(self.k) if t != self.bos_t and t != self.eos_t]
    

        self.alpha = [torch.full((self.k,), float('-inf')) for _ in isent]
        self.alpha[0] = torch.log(self.eye[self.bos_t])

        # Forward pass - same approach as the viterbi algo w slight chnages 
        for j in range(1, n +1):
            word_id= isent[j][0]
            
            for t in valid_tags:  
                for s in valid_tags: 
                    if self.A[s,t] > 0 and self.B[t,word_id] > 0:
                        term = (self.alpha[j-1][s] + 
                            torch.log(self.A[s,t]) + 
                            torch.log(self.B[t,word_id]))
                        
                        # adding in log space
                        if self.alpha[j][t] == float('-inf'):
                            self.alpha[j][t] = term
                        else:
                            self.alpha[j][t] = torch.logsumexp(
                                torch.tensor([self.alpha[j][t], term]), dim=0)
        
        # similar as well to Viterbis, handling the EOS transition
        for s in valid_tags:
            if self.A[s,self.eos_t] > 0:
                term = self.alpha[n][s] + torch.log(self.A[s,self.eos_t])
                if self.alpha[n+1][self.eos_t] == float('-inf'):
                    self.alpha[n+1][self.eos_t] = term
                else:
                    self.alpha[n+1][self.eos_t] = torch.logsumexp(
                        torch.tensor([self.alpha[n+1][self.eos_t], term]), dim=0)
    
        # reps the probability of getting to EOS
        self.log_Z = self.alpha[-1][self.eos_t]

            # Note: once you have this working on the ice cream data, you may
            # have to modify this design slightly to avoid underflow on the
            # English tagging data. See section C in the reading handout.

        return self.log_Z

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """Run the backwards algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the backward
        probability). 
        
        As a side effect, add the expected transition and emission counts (times
        mult) into self.A_counts and self.B_counts.  These depend on the alpha
        values and log Z, which were stored for us (in self) by the forward
        pass."""

        # Pre-allocate beta just as we pre-allocated alpha.
        beta = [torch.empty(self.k) for _ in isent]
        beta[-1] = self.eye[self.eos_t]  # vector that is one-hot at EOS_TAG
       
        raise NotImplementedError   # you fill this in!

        return log_Z_backward

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""


        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence. You'll have to
        # deintegerize the words and tags again when constructing the return
        # value, since the type annotation on this method says that it returns a
        # Sentence object, and that's what downstream methods like eval_tagging
        # will expect.  (Running mypy on your code will check that your code
        # conforms to the type annotations ...)

        isent = self._integerize_sentence(sentence, corpus)
        n = len(isent) - 2  # exclude BOS and EOS
        
        # exclusing BOS and EOS 
        valid_tags = [t for t in range(self.k) if t != self.bos_t and t != self.eos_t]

        alpha = [torch.full((self.k,), float('-inf')) for _ in range(n + 2)]
        backpointers = [torch.full((self.k,), -1, dtype=torch.int) for _ in range(n + 2)]

        # (log probability of 1)
        alpha[0][self.bos_t] = 0

        # for position 1, first word after BOS
        word_id = isent[1][0]
        for t in valid_tags:
            if self.A[self.bos_t, t] > 0 and self.B[t, word_id] > 0:
                alpha[1][t] = alpha[0][self.bos_t] + torch.log(self.A[self.bos_t, t]) + torch.log(self.B[t, word_id])
                backpointers[1][t] = self.bos_t
            else:
                # for zero probabilities
                alpha[1][t] = float('-inf')
                backpointers[1][t] = -1

        # Forward pass
        for j in range(2, n + 1):  # Positions 2 to n
            word_id = isent[j][0]
            for t in valid_tags:
                max_score = float('-inf')
                best_s = -1
                for s in valid_tags:
                    if self.A[s, t] > 0 and self.B[t, word_id] > 0 and alpha[j - 1][s] > float('-inf'):
                        score = alpha[j - 1][s] + torch.log(self.A[s, t]) + torch.log(self.B[t, word_id])
                        if score > max_score:
                            max_score = score
                            best_s = s
                alpha[j][t] = max_score
                backpointers[j][t] = best_s

        # transition to EOS at position n+1
        max_score = float('-inf')
        best_s = -1
        for s in valid_tags:
            if self.A[s, self.eos_t] > 0 and alpha[n][s] > float('-inf'):
                score = alpha[n][s] + torch.log(self.A[s, self.eos_t])
                if score > max_score:
                    max_score = score
                    best_s = s
        alpha[n + 1][self.eos_t] = max_score
        backpointers[n + 1][self.eos_t] = best_s

        # Backtracking
        tags = []
        current_tag = self.eos_t
        for j in range(n + 1, 0, -1): 
            prev_tag = backpointers[j][current_tag]
            if prev_tag == -1:
                raise ValueError(f"No valid path at position {j}")
            if j != 0 and current_tag != self.eos_t and current_tag != self.bos_t:
                tags.insert(0, current_tag)
            current_tag = prev_tag

        # now include BOS and EOS
        result = []
        tags_index = 0  
        for j, (word, _) in enumerate(sentence):
            if j == 0:  # BOS
                result.append((word, BOS_TAG))
            elif j == len(sentence) - 1:  # EOS
                result.append((word, EOS_TAG))
            else:
                result.append((word, self.tagset[tags[tags_index]]))
                tags_index += 1
        return Sentence(result)
            


    def save(self, model_path: Path) -> None:
        logger.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> HiddenMarkovModel:
        model = torch.load(model_path, map_location=device)\
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        if model.__class__ != cls:
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {model_path}.")

        logger.info(f"Loaded model from {model_path}")
        return model
