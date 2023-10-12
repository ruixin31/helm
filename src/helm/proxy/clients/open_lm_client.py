import os
import glob
import json
import logging
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Literal, Optional, Dict, Union
from yaml import Loader

import torch
from dataclasses import asdict, fields

from helm.common.cache import Cache, CacheConfig, KeyValueStoreCacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)

from helm.proxy.clients.client import Client, wrap_request_time, truncate_sequence, cleanup_tokens

try:
    from open_lm.model import Transformer, create_model, Params
    from transformers import GPTNeoXTokenizerFast
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# From Lit_GPT integration (#1792) to save gpu memory
class SingletonMeta(type):
    _instances: Dict[type, type] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class GenerationArgs:
    max_gen_len: int = 200
    temperature: float = 0.8
    top_p: float = 0.95
    # need to investigate further
    top_k_per_token: int = 1
    num_return_sequences: int = 1


class Generator:
    def __init__(self, model: Transformer):
        self.model = model
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        self.pad_token_id = 50282
        self.seq_len = 2048

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        gen_args: GenerationArgs = GenerationArgs(),
    ) -> List[str]:
        bsz = len(prompts)

        prompt_tokens = [self.tokenizer.encode(x) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(self.seq_len, gen_args.max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_token_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_token_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            last_logits = self.model(tokens[:, prev_pos:cur_pos].clone())[0][:, -1, :]
            if gen_args.temperature > 0:
                probs = torch.softmax(last_logits / gen_args.temperature, dim=-1)
                next_token = sample_top_p(probs, gen_args.top_p)
            else:
                next_token = torch.argmax(last_logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # TODO: enable caching again for inference
            # prev_pos = cur_pos

        # For comptuing log_probs
        scores = self.model(tokens.clone())[0]
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []

        for completion_id in range(gen_args.num_return_sequences):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            # for i in range(len(sequences[completion_id])):
            for i in range(len(tokens[completion_id]) - 1):
                logprobs = torch.nn.functional.log_softmax(scores[completion_id][i], dim=0)
                topk_logprobs = torch.topk(logprobs, k=gen_args.top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )
                # logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][i + 1]].item())
                logprobs_of_chosen_tokens.append(logprobs[tokens[completion_id][i + 1]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # all_decoded_text =
        all_tokens = [[self.tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in tokens]
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = t[: len(prompt_tokens[i]) + gen_args.max_gen_len]
            decoded_i = self.tokenizer.decode(t)

            decoded.append(decoded_i)
            # decoded = []
            # for t in decoded_i:
            #     decoded.append(t)

        # one completion
        # Need truncate
        return decoded, all_logprobs_of_chosen_tokens, all_tokens, all_top_logprobs_dicts


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def create_params(path: str):
    with open(path, "r") as f:
        # params = yaml.load(f, Loader=Loader)
        params = json.load(f)
    params["dim"] = params["hidden_dim"]
    del params["hidden_dim"]

    return Params(**params)


class ModelArgs:
    def __init__(self, path: str):
        with open(path, "r") as f:
            # params = yaml.load(f, Loader=Loader)
            params = json.load(f)
        params["dim"] = params["hidden_dim"]
        params_obj = Params(**params)
        # reutrn params_obj
        for field in fields(Params):
            setattr(self, field.name, getattr(params, field.name))

        # for k, v in params.items():
        #     setattr(self, k, v)


class OpenLM(metaclass=SingletonMeta):
    def __init__(
        self,
        checkpoint_path: Optional[Path] = Path(""),
        params_path: Optional[Path] = Path(""),
        wandb_dir: Optional[Path] = Path(""),
    ):
        if wandb_dir.exists():
            if not params_path.exists():
                params_path = wandb_dir / "params.txt"
            if not checkpoint_path.exists():
                chkpt_dir = wandb_dir / "checkpoints" / "epoch_*.pt"
                list_of_files = glob.glob(chkpt_dir)
                latest_file = max(list_of_files, key=os.path.getctime)
                checkpoint_path = latest_file
        else:
            assert params_path.exists(), "Must provide params file or a wandb directory."
            assert checkpoint_path.exists(), "Must provide checkpoint file or a wandb directory."

        # open_lm = create_model(ModelArgs(params_path)).half()
        # open_lm = Transformer(create_params(params_path)).half()
        open_lm = Transformer(create_params(params_path))
        checkpoint = torch.load(checkpoint_path)

        state_dict = checkpoint["state_dict"]
        state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
        open_lm.load_state_dict(state_dict)
        # Can change device here
        open_lm.eval().cuda()
        generator = Generator(open_lm)

        self.model = open_lm
        self.generator = generator
        self.tokenizer = generator.tokenizer


class OpenLMClient(Client):
    """Client for evaluating OpenLM supported LLMs"""

    def __init__(
        self,
        cache_config: CacheConfig,
        checkpoint_dir: Optional[Path] = Path(""),
        params_dir: Optional[Path] = Path(""),
        wandb_dir: Optional[Path] = Path(""),
    ):
        self.cache = Cache(cache_config)
        open_lm = OpenLM(checkpoint_dir, params_dir, wandb_dir)
        self.model = open_lm.model
        self.generator = open_lm.generator
        self.tokenizer = open_lm.tokenizer

    def make_request(self, request: Request) -> RequestResult:
        generate = self.generator.generate
        # tokenizer = self.tokenizer
        # fabric = self.fabric
        input_text = [request.prompt]
        decoded, all_logprobs_of_chosen_tokens, all_tokens, all_top_logprobs_dicts = generate(
            input_text,
            GenerationArgs(request.max_tokens, request.temperature, request.top_p),
        )

        completions = []
        for i, sequence_text in enumerate(decoded):
            sequence_logprob: float = 0
            tokens: List[Token] = []
            for token_text, logprob, top_logprobs_dict in zip(
                all_tokens[i], all_logprobs_of_chosen_tokens[i], all_top_logprobs_dicts[i]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob
            completion = Sequence(text=sequence_text, logprob=sequence_logprob, tokens=tokens)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=False,
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = self.tokenizer
        t0 = time.perf_counter()

        # try:

        # def do_it():
        if request.encode:
            if request.truncation:
                tokens = tokenizer.encode(
                    request.text,
                    truncation=request.truncation,
                    max_length=request.max_length,
                    add_special_tokens=False,
                )
            else:
                tokens = tokenizer.encode(request.text, add_special_tokens=False)
        else:
            if "gpt" in request.tokenizer or request.tokenizer in [
                "bigscience/bloom",
                "Writer/palmyra-base",
                "facebook/opt-66b",
            ]:
                # These models already handle the "▁" character correctly with the
                # convert_tokens_to_string method. We prefer to use this method instead
                # of the hacky cleanup_tokens method below as it might handle cases
                # we haven't thought of in cleanup_tokens.
                tokens = [tokenizer.convert_tokens_to_string([token]) for token in tokenizer.tokenize(request.text)]
            else:
                # Tokenizes the text and returns the tokens as a list of strings,
                # not a list of token objects (otherwise "Hello world" would be"
                # ["Hello", "▁world"] and not ["Hello", " world"])
                # We could do this with a simple replace like this:
                # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                # But this replaces all the "▁" characters by "", which is not what we want.
                # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                # Just like tokenize("Hello", encode=False) would return ["Hello"].
                tokens = tokenizer.tokenize(request.text)
                tokens = cleanup_tokens(tokens, request.tokenizer)

        # result, cached = self.cache.get(cache_key, wrap_request_time(do_it))

        # except Exception as e:
        #     error: str = f"HuggingFace error: {e}"
        #     return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])
        t = time.perf_counter() - t0

        return TokenizationRequestResult(
            success=True,
            cached=False,
            text=request.text,
            tokens=[TokenizationToken(value) for value in tokens],
            request_time=t,
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = self.tokenizer
        t0 = time.perf_counter()

        result = {
            "text": tokenizer.decode(request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces)
        }

        t = time.perf_counter() - t0

        return DecodeRequestResult(success=True, cached=False, text=result["text"], request_time=t)

