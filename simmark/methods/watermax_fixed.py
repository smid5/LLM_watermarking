import torch
import numpy as np
from scipy.stats import norm
# NOTE:
import hashlib, hmac

from .seeding import simhash_seed, normal_seed

def _sample_top_p_candidates(probs, top_p, num_samples):
    # return up to num_samples token indices sampled from the top_p distribution
    # probs: 1d tensor of size vocab_size
    # top_p: float in (0,1) (nucleus threshold)
    # num_samples: int, number of samples to draw (with replacement)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # now we find cuffoff index where cumulative_probs exceeds top_p
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False).item()
    # clamp to avoid an out-of-range on top_p about 1.0
    cutoff_index = min(cutoff_index, probs.numel() - 1)

    # keep tokens up to and including cutoff_index
    top_p_sorted_indices = sorted_indices[:cutoff_index+1]
    top_p_probs = probs[top_p_sorted_indices]
    # renormalize
    top_p_probs = top_p_probs / top_p_probs.sum()  

    # lets add a numerical stability check
    kept_probs = torch.where(torch.isfinite(top_p_probs), top_p_probs, torch.tensor(0.0, device=top_p_probs.device, dtype=top_p_probs.dtype))
    if kept_probs.sum() <= 0:
        # pick single most probable token
        return top_p_sorted_indices[:1]
    
    draws = torch.multinomial(kept_probs, num_samples=num_samples, replacement=True)
    return top_p_sorted_indices[draws]

def _hmac64(salt_int, prefix_key_int, token_int):
    # NOTE: co-pilot seems to think this is correctly implremented, but we should check :)
    salt = salt_int.to_bytes(8, 'big', signed=False)
    msg = prefix_key_int.to_bytes(8, 'big', signed=False) + int(token_int).to_bytes(8, 'big', signed=False)
    mac = hmac.new(salt, msg, hashlib.sha256).digest()
    return int.from_bytes(mac[:8], 'big', signed=False)

class WaterMaxProcessor(torch.nn.Module):
    # token level low latenxy (m=1) watermax implementation, allows simhas seeding
    # at a high level, this is how it works
    # 1. at each gen step, propose num_drafts token candidates from top_p nucleus
    # 2. for each candidate, compute deterministic seed from the causal prefix including
    #    the candidate token using the seed function (which can be simhash or normal or whatever)
    # 3. seed the prng and sample a random variable u from N(0,1)
    # 4. select the candidate with the largest random variable u (i.e. smallest one step p value) 
    #    and force that token
    # some comments: per position random variable is iid under H0 (i.e. the first prefix)
    # and is seeded by a function of the current tokens and preceding h-1 tokens (hash window
    # size is h). so the random variable is deterministic function of the causal prefix
    # also, to preserve token score independence, we discard repeated n grams. so we keep a 
    # "seen signature" set of all ngrams in the current sequence for candidate token, 
    # to downweight its r.v. if the candidate token creates a repeated ngram
    
    def __init__(self, generation_config):
        super().__init__()
        self.model = generation_config['model']
        self.vocab_size = generation_config['vocab_size']

        # base seed for simhash
        self.seed = generation_config['seed']  

        self.transformer_model = generation_config['transformer_model']
        self.tokenizer = generation_config['tokenizer']
        self.prior_tokens = generation_config['prior_tokens']

        # simhash hash size
        self.k = generation_config['k'] 
        # simhash band size 
        self.b = generation_config['b']  

        self.seed_function = generation_config.get('seed_function', simhash_seed)
        # nucleus sampling threshold
        self.top_p = generation_config.get('top_p', 0.9)  
        # number of candidate tokens to propose
        self.num_drafts = generation_config.get('num_drafts', 8)  

        self.seen_signatures = set()
        
    def forward(self, input_ids, logits):
        # ok, so we will choose next token by maximizing the watermax score increment
        # for each batch row, we will
        # 1. build small token candidate set by sampling from top_p nucleus
        # 2. for each candidate, compute seed from causal prefix + candidate
        # 3. seed prng and sample u ~ N(0,1) (penalize if necessary)
        # 4. select candidate with largest u (smallest p value) and force that token

        batch_size = input_ids.shape[0]
        device = logits.device
        for batch in range(batch_size):
            # sample candidate set from top_p nucleus
            probs = logits[batch].softmax(dim=-1)

            # 1. propose candidates
            candidate_ids = _sample_top_p_candidates(probs, self.top_p, self.num_drafts)
            candidate_ids = candidate_ids.to(device)

            # NOTE: NOW, we are going tocompute a prefix key K_t 
            # from the prefix ONLY (no candidate)
            prefix_ids = input_ids[batch]  
            K_t = self.seed_function(prefix_ids,
                                     self.prior_tokens,
                                     self.tokenizer,
                                     self.transformer_model,
                                     hash_idx=0,
                                     seed=self.seed,
                                     k=self.k,
                                     b=self.b)
            # dedup prefix key
            signature = K_t 

            best_u = -np.inf
            best_token = candidate_ids[0].item()
            # record signature for dedup
            best_signature = signature  

            for c in candidate_ids.tolist():
                # 2. compute seed from causal prefix + candidate
                #    NOTE: NOW, we derive candidate-specific seed via HMAC(K_t, c))
                cand_seed = _hmac64(self.seed, K_t, int(c))
                rng = np.random.default_rng(cand_seed)
                # sample u ~ N(0,1)
                u = rng.standard_normal() 

                # mirror watermax's dedup: repeated windows contribute zero to S
                # if signature in self.seen_signatures:
                #     # NOTE: we could also subtract a small penalty here instead of zeroing out
                #     u = 0.0

                if u > best_u:
                    best_u = u
                    best_token = c
                    # best_signature remains the prefix signature (used for dedup)

            # record the chosen signature to enforce per-text independence
            # self.seen_signatures.add(best_signature)

            # finally, force best token
            logits[batch,:] = 1e-5
            logits[batch, best_token] = 1e5

        return logits
    

def watermax_detect(text, config):
    # we will detect watermax simhash watermark by computing the simhash
    # 1. for each token i, define a seeded variable u_i ~ N(0,1) 
    #   where the seed is a function of thecurrent token and preceding h-1 tokens
    # 2. the global score is S = sum_i u_i, over subset I that excludes dups
    # 3. let M = |I|, then under H0, s ~ N(0,M) so we can compute p value as
    #   p = Phi( -S / sqrt(M) )
    # we'll obvs reject for some small p value threshold
    tokenizer = config['tokenizer']
    transformer_model = config['transformer_model']
    prior_tokens = config['prior_tokens']
    seed = config['seed']
    k = config['k']
    b = config['b']
    seed_function = config.get('seed_function', simhash_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ids = tokenizer.encode(text, return_tensors="pt").squeeze().to(device)

    S = 0.0
    M = 0
    # seen_signatures = set()

    # so now we iterate over tokens, for each position i, we build causal
    # prefix that incluses the current token and pass the seed function, 
    # (internally this truncates the last prior token) and we dedup on the
    # sig
    for i in range(len(ids)):
        # prefix ONLY for the key (exclude current token)
        prefix_ids = ids[:i]
        # handle i==0 (empty prefix) by falling back to just current token context
        if prefix_ids.numel() == 0:
            prefix_ids = ids[:1]

        K_t = seed_function(prefix_ids, 
                            prior_tokens, 
                            tokenizer, 
                            transformer_model, 
                            hash_idx=0, 
                            seed=seed, 
                            k=k, 
                            b=b)
        # if K_t in seen_signatures:
        #     continue
        # seen_signatures.add(K_t)

        # derive candidate-specific seed with the *observed* token id ids[i]
        cand_seed = _hmac64(seed, K_t, int(ids[i].item()))
        rng = np.random.default_rng(cand_seed)
        u = rng.standard_normal()
        S += float(u)
        M += 1

    if M <= 0:
        print("warning, M <= 0 in watermax detection, no independent tokens found, returning p=0.5")
        return 0.5
    
    z = S / np.sqrt(M)
    p_value = float(norm.cdf(-z))
    print(f"watermax simhash detect: S = {S}, M = {M}, z = {z}, p = {p_value}")
    return p_value