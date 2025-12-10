import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import datasets
import numpy as np
import torch
import tqdm
import transformers
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

datasets.disable_caching()


def emb(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        emb = model.call_embedding_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    return emb


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def get_num_proc() -> int:
    world_size: int = get_world_size()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size


def embed_all_tokens(model: torch.nn.Module, tokenizer: transformers.AutoTokenizer):
    """Generates embeddings for all tokens in tokenizer vocab."""
    i = 0
    model.embedder.eval()
    batch_size = 1024
    all_token_embeddings = []
    V = tokenizer.vocab_size
    #
    # DPR has CLS and SEP.
    # GTR has no CLS or start token at all, and has EOS at the end.
    CLS = tokenizer.cls_token_id
    SEP = (tokenizer.sep_token_id) or (tokenizer.eos_token_id)
    assert SEP is not None
    #
    device = next(model.parameters()).device
    pbar = tqdm.tqdm(
        desc="generating token embeddings", colour="#008080", total=V, leave=False
    )
    while i < V:
        #
        minibatch_size = min(V - i, batch_size)
        inputs = torch.arange(i, min(i + minibatch_size, V))
        #
        if CLS is not None:
            input_ids = torch.stack(
                [
                    torch.tensor([CLS]).repeat(len(inputs)),
                    inputs,
                    torch.tensor([SEP]).repeat(len(inputs)),
                ]
            ).T
        else:
            input_ids = torch.stack([inputs, torch.tensor([SEP]).repeat(len(inputs))]).T
        input_ids = input_ids.to(device)
        #
        attention_mask = torch.ones_like(input_ids, device=device)
        #
        with torch.no_grad():
            token_embeddings = emb(model, input_ids, attention_mask)
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    #
    all_token_embeddings_tensor: torch.Tensor = torch.stack(all_token_embeddings)
    assert all_token_embeddings_tensor.shape == (tokenizer.vocab_size, 768)

    all_token_embeddings_tensor /= all_token_embeddings_tensor.norm(
        p=2, dim=1, keepdim=True
    )
    return all_token_embeddings_tensor


def torch_main_worker_finish_first(func: Callable):
    def wrapper(*args, **kwargs):
        # Get local rank (need to support non-DDP).
        try:
            local_rank = torch.distributed.get_rank()
            ddp_enabled = True
        except (RuntimeError, ValueError):
            local_rank = -1
            ddp_enabled = False
        is_main_worker = local_rank <= 0
        # Run on main worker first.
        if is_main_worker:
            result = func(*args, **kwargs)
        # Then everyone waits.
        if ddp_enabled:
            torch.distributed.barrier()
        # Run on other workers now.
        if not is_main_worker:
            result = func(*args, **kwargs)
        # Now everyone waits again.
        if ddp_enabled:
            torch.distributed.barrier()
        return result

    return wrapper


def dataset_map_multi_worker(
    dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs
) -> datasets.Dataset:

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
    except (RuntimeError, ValueError):
        # In non-distributed mode, just run regular map()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
        return dataset.map(map_fn, *args, **kwargs)
    datasets.disable_caching()

    cache_path = os.environ.get(
        "VEC2TEXT_CACHE", os.path.expanduser("~/.cache/inversion")
    )
    
    # Debug: Comprehensive disk and environment checks
    import shutil as disk_utils
    import tempfile
    
    print(f"[DEBUG] rank {rank}: ============ STARTING DATASET_MAP_MULTI_WORKER DEBUG ============")
    print(f"[DEBUG] rank {rank}: Process PID: {os.getpid()}")
    print(f"[DEBUG] rank {rank}: Current working directory: {os.getcwd()}")
    print(f"[DEBUG] rank {rank}: cache_path={cache_path}")
    print(f"[DEBUG] rank {rank}: Cache path exists: {os.path.exists(cache_path)}")
    
    # Check all relevant disk spaces
    for path_name, path in [("cache_path", cache_path), ("home", os.path.expanduser("~")), ("tmp", "/tmp"), ("current_dir", os.getcwd())]:
        try:
            if os.path.exists(path):
                disk_usage = disk_utils.disk_usage(path)
                print(f"[DEBUG] rank {rank}: {path_name} ({path}) - total={disk_usage.total/1024**3:.1f}GB, used={disk_usage.used/1024**3:.1f}GB, free={disk_usage.free/1024**3:.1f}GB")
            else:
                print(f"[DEBUG] rank {rank}: {path_name} ({path}) - PATH DOES NOT EXIST")
        except Exception as e:
            print(f"[DEBUG] rank {rank}: Error checking {path_name}: {e}")
    
    # Check environment variables
    env_vars = ["TMPDIR", "TMP", "TEMP", "HF_DATASETS_CACHE", "HF_HOME", "VEC2TEXT_CACHE"]
    for var in env_vars:
        value = os.environ.get(var, "NOT_SET")
        print(f"[DEBUG] rank {rank}: ENV {var}={value}")
    
    # Check memory usage
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"[DEBUG] rank {rank}: Memory - total={memory_info.total/1024**3:.1f}GB, available={memory_info.available/1024**3:.1f}GB, used={memory_info.used/1024**3:.1f}GB")
    except ImportError:
        print(f"[DEBUG] rank {rank}: psutil not available, skipping memory check")
    except Exception as e:
        print(f"[DEBUG] rank {rank}: Error checking memory: {e}")
    
    # Check temp directory
    temp_dir = tempfile.gettempdir()
    print(f"[DEBUG] rank {rank}: Python tempfile.gettempdir() = {temp_dir}")
    
    disk_usage = disk_utils.disk_usage(cache_path)
    print(f"[DEBUG] rank {rank}: INITIAL cache disk_usage - total={disk_usage.total/1024**3:.1f}GB, used={disk_usage.used/1024**3:.1f}GB, free={disk_usage.free/1024**3:.1f}GB")
    
    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]
    print(f"[DEBUG] rank {rank}: Target shard filepath: {ds_shard_filepaths[rank]}")
    print(f"[DEBUG] rank {rank}: Dataset original size: {len(dataset)}")
    
    # Create dataset shard
    print(f"[DEBUG] rank {rank}: Creating dataset shard...")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    print(f"[DEBUG] rank {rank}: Shard size: {len(ds_shard)}")
    
    # Debug: Check disk space before mapping
    disk_usage_before = disk_utils.disk_usage(cache_path)
    print(f"[DEBUG] rank {rank}: Before mapping - free={disk_usage_before.free/1024**3:.1f}GB")
    
    # Check if we have temp directory space
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_usage = disk_utils.disk_usage(temp_dir)
    print(f"[DEBUG] rank {rank}: temp_dir={temp_dir}, temp_free={temp_usage.free/1024**3:.1f}GB")
    
    # Check datasets cache directory
    try:
        datasets_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        if os.path.exists(datasets_cache):
            datasets_usage = disk_utils.disk_usage(datasets_cache)
            print(f"[DEBUG] rank {rank}: datasets_cache={datasets_cache}, free={datasets_usage.free/1024**3:.1f}GB")
        else:
            print(f"[DEBUG] rank {rank}: datasets_cache={datasets_cache} does not exist")
    except Exception as e:
        print(f"[DEBUG] rank {rank}: Error checking datasets cache: {e}")
    
    print(f"[DEBUG] rank {rank}: Starting dataset.map() operation...")
    print(f"[DEBUG] rank {rank}: map_fn={map_fn}, args={args}, kwargs keys={list(kwargs.keys())}")
    
    try:
        ds_shard = ds_shard.map(map_fn, *args, **kwargs)
        print(f"[DEBUG] rank {rank}: dataset.map() completed successfully")
    except Exception as e:
        print(f"[DEBUG] rank {rank}: dataset.map() FAILED with error: {e}")
        print(f"[DEBUG] rank {rank}: Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Debug: Check disk space after mapping
    disk_usage_after = disk_utils.disk_usage(cache_path)
    print(f"[DEBUG] rank {rank}: After mapping - free={disk_usage_after.free/1024**3:.1f}GB")
    
    print(f"[DEBUG] rank {rank}: Starting save_to_disk operation...")
    try:
        ds_shard.save_to_disk(ds_shard_filepaths[rank])
        print(f"[DEBUG] rank {rank}: save_to_disk completed successfully to {ds_shard_filepaths[rank]}")
        
        # Check size of saved files
        if os.path.exists(ds_shard_filepaths[rank]):
            try:
                import subprocess
                result = subprocess.run(['du', '-sh', ds_shard_filepaths[rank]], capture_output=True, text=True)
                print(f"[DEBUG] rank {rank}: Saved cache size: {result.stdout.strip()}")
            except Exception as e:
                print(f"[DEBUG] rank {rank}: Could not get cache size: {e}")
        
    except Exception as e:
        print(f"[DEBUG] rank {rank}: save_to_disk FAILED with error: {e}")
        print(f"[DEBUG] rank {rank}: Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise
    print(f"[DEBUG] rank {rank}: Waiting at barrier before dataset concatenation...")
    torch.distributed.barrier()
    
    print(f"[DEBUG] rank {rank}: Loading datasets for concatenation...")
    try:
        loaded_datasets = []
        for i, p in enumerate(ds_shard_filepaths):
            print(f"[DEBUG] rank {rank}: Loading dataset {i} from {p}")
            if os.path.exists(p):
                loaded_ds = datasets.load_from_disk(p)
                print(f"[DEBUG] rank {rank}: Loaded dataset {i}, size: {len(loaded_ds)}")
                loaded_datasets.append(loaded_ds)
            else:
                print(f"[DEBUG] rank {rank}: ERROR: Dataset path {p} does not exist!")
                
        full_dataset = datasets.concatenate_datasets(loaded_datasets)
        print(f"[DEBUG] rank {rank}: Concatenated dataset size: {len(full_dataset)}")
    except Exception as e:
        print(f"[DEBUG] rank {rank}: Dataset concatenation FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"[DEBUG] rank {rank}: Waiting at barrier after dataset concatenation...")
    torch.distributed.barrier()
    
    # Staggered cleanup to avoid race conditions
    cleanup_delay = rank * 0.5  # rank 0: 0s, rank 1: 0.5s, rank 2: 1s, etc.
    print(f"[DEBUG] rank {rank}: Starting cleanup with {cleanup_delay}s delay...")
    time.sleep(cleanup_delay)
    
    print(f"[DEBUG] rank {rank}: Beginning cleanup of {ds_shard_filepaths[rank]}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if os.path.exists(ds_shard_filepaths[rank]):
                print(f"[DEBUG] rank {rank}: Cleanup attempt {attempt + 1}: removing {ds_shard_filepaths[rank]}")
                shutil.rmtree(ds_shard_filepaths[rank])
                print(f"[DEBUG] rank {rank}: Cleanup successful on attempt {attempt + 1}")
                break
            else:
                print(f"[DEBUG] rank {rank}: Path {ds_shard_filepaths[rank]} already removed")
                break
        except OSError as e:
            print(f"[DEBUG] rank {rank}: Cleanup attempt {attempt + 1} failed: {e} (errno: {e.errno})")
            if e.errno == 39 and attempt < max_retries - 1:  # Directory not empty
                print(f"[DEBUG] rank {rank}: Directory not empty, retrying in 1s...")
                time.sleep(1)
            else:
                print(f"[DEBUG] rank {rank}: Cleanup failed after {max_retries} attempts: {e}")
                # Continue without failing the entire training
                break
        except Exception as e:
            print(f"[DEBUG] rank {rank}: Unexpected cleanup error: {e}")
            break
    
    print(f"[DEBUG] rank {rank}: ============ DATASET_MAP_MULTI_WORKER COMPLETE ============")
    
    return full_dataset


manifest_object = None


def get_manifest_global():
    from manifest import Manifest

    global manifest_object
    if manifest_object is None:
        manifest_object = Manifest(
            client_name="openaiembedding",  # defaults to 'text-embedding-ada-002'
            # cache_name="sqlite",
            # cache_connection="/home/jxm3/.manifest/jxm_openai_manifest.sqlite",
        )
        # manifest_object.PARAMS = {
        #     'engine': ('model', 'text-embedding-ada-002'),
        #     'batch_size': ('batch_size', 128),
        # }
    return manifest_object


@retry(wait=wait_fixed(1), stop=stop_after_attempt(15))
def get_embeddings_openai_manifest(
    text_list, model="text-embedding-ada-002"
) -> np.ndarray:
    # embeddings model: https://platform.openai.com/docs/guides/embeddings/use-cases
    #    api ref: https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: set up a caching system somehow.
    manifest = get_manifest_global()
    # print(
    #     f"running manifest on text_list of length {len(text_list)}, first element '{text_list[0]}'"
    # )
    return np.array(manifest.run(text_list, batch_size=min(len(text_list), 128)))


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def get_embeddings_openai_vanilla_multithread(
    text_list, model="text-embedding-ada-002"
) -> list:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = OpenAI(api_key=api_key)

    # print(f"running openai on text_list of length {len(text_list)}, first element '{text_list[0]}'")

    batches = math.ceil(len(text_list) / 128)
    outputs = []

    for i in range(len(text_list)):
        if len(text_list[i]) == 0:
            print(f"warning: set element {i} to a random sequence")
            text_list[i] = "random sequence"

    def process_batch(batch):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch, model=model, encoding_format="float"
        )
        return [e.embedding for e in response.data]

    with ThreadPoolExecutor() as executor:
        batch_indices = range(batches)
        results = executor.map(process_batch, batch_indices)

        for result in results:
            outputs.extend(result)

    return outputs


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def get_embeddings_openai_vanilla(text_list, model="text-embedding-ada-002") -> list:
    # embeddings model: https://platform.openai.com/docs/guides/embeddings/use-cases
    #    api ref: https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: set up a caching system somehow.
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = OpenAI(api_key=api_key)

    # print(f"running openai on text_list of length {len(text_list)}, first element '{text_list[0]}'")
    batches = math.ceil(len(text_list) / 128)
    outputs = []
    for batch in range(batches):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch, model=model, encoding_format="float"
        )
        outputs.extend([e.embedding for e in response.data])
    return outputs


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def embed_api(
    input_ids: torch.Tensor,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    api_name: str,
) -> torch.Tensor:
    text_list = embedder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    # get_embeddings_func = get_embeddings_openai_vanilla
    get_embeddings_func = get_embeddings_openai_vanilla_multithread
    # get_embeddings_func = get_embeddings_openai_manifest
    if api_name.startswith("text-embedding-ada") or api_name.startswith("text-embedding-3"):
        embeddings = get_embeddings_func(
            text_list=text_list,
            model=api_name,
        )
    else:
        raise ValueError(f"unsupported api name {api_name}")

    return torch.tensor(embeddings, device=input_ids.device, dtype=torch.float32)


class MockEmbedder:
    embedder_dim: int

    def __init__(self, embedder_dim: int):
        self.embedder_dim = embedder_dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )
