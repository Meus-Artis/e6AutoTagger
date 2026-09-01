import argparse
import os
import sys
import base64
from asyncio import Future, get_running_loop, to_thread
from dataclasses import dataclass
from functools import lru_cache
from queue import Empty, SimpleQueue
from threading import Thread
from typing import Any, Annotated, Iterable
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
import torch
from torch import Tensor
import uvicorn
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse
from pydantic import RootModel, BaseModel, Field
from hydra import image
from hydra.classification import IMPLICATION_MODES, Calibration
from hydra.model import Hydra, Extension, load_model
from utils.workqueue import WorkQueue
from huggingface_hub import hf_hub_download
import shutil
import time
START_TIME = time.time()
REQUEST_COUNT = 0
EXCLUDED_TAGS = frozenset({"safe","questionable","explicit","busty_boy_(lore)","female_(lore)","male_(lore)","sibling_(lore)","trans_man_(lore)","trans_woman_(lore)"})
print("Checking for model...")
if not os.path.isfile("JTP-Hydra-3.5.safetensors"):
    print("Downloading tagger model, please wait...")
    path = hf_hub_download(repo_id="RedRocket/Hydra", filename="models/hydra-3.5.safetensors", repo_type="model", local_dir="./")
    shutil.move(path, "JTP-Hydra-3.5.safetensors")
    os.rmdir("models")
else:
    print("JTP-Hydra-3.5.safetensors found")
@dataclass
class Work:
    image: Tensor
    calibration: Calibration
    implications: str
    future: Future | None
    def take_image(self) -> Tensor:
        image = self.image
        del self.image
        return image
    def finish(self, output: Tensor) -> None:
        try:
            result = self.calibration.classify_output(output, implications=self.implications)
        except Exception as ex:
            self.error(ex)
        else:
            assert self.future is not None
            self.future.get_loop().call_soon_threadsafe(self.future.set_result, result)
            self.future = None
    def error(self, ex: BaseException) -> None:
        if self.future is not None:
            self.future.get_loop().call_soon_threadsafe(self.future.set_exception, ex)

class ClassifyResponse(RootModel[dict[
    Annotated[str, Field(description="label")],
    Annotated[float, Field(description="probability", ge=0.0, le=1.0)]
]]):
    pass

class LabelInfo(BaseModel):
    category: Annotated[str, Field(description="Label category.")]
    subcategory: Annotated[str | None, Field(description="Label subcategory.")]
    implies: Annotated[list[str], Field(description="List of implied labels.")]

class ImageConfig(BaseModel):
    colorspace: str
    background: list[int]
    resize_kernel: str
    resize_linear: bool
    patch_size: int
    max_seqlen: int

class InfoResponse(BaseModel):
    model: Annotated[str, Field(description="Model name.")]
    extensions: Annotated[list[str], Field(description="Filenames of all loaded extensions.")]
    image_config: Annotated[ImageConfig, Field(description="Native image format information.")]
    labels: Annotated[dict[
        Annotated[str, Field(description="label")],
        LabelInfo
    ], Field(description="Metadata for all supported labels.")]

APP = FastAPI(title="Hydra Classifier")
MODEL: Hydra
EXTS: list[str] = []
ARGS: argparse.Namespace
QUEUE: SimpleQueue[Work] = SimpleQueue()
WQ = WorkQueue(name="cpu-worker", daemon=True)

@lru_cache(maxsize=16)
def get_calibration(metric: str) -> Calibration:
    if metric == "none":
        return MODEL.calibrate(0.0)
    return MODEL.calibrate(metric)

@torch.inference_mode()
def handle_batch(batch: list[Work], outputs: Tensor) -> None:
    try:
        outputs = outputs.cpu()
        for work, output in zip(batch, outputs.unbind()):
            work.finish(output)
    except Exception as ex:
        for work in batch:
            work.error(ex)

@torch.inference_mode()
def worker() -> None:
    device = MODEL.embeds.pos_embed.device
    seqlen = MODEL.image_config.max_seqlen
    while True:
        batch = [QUEUE.get()]
        while len(batch) < ARGS.batch_size:
            try:
                batch.append(QUEUE.get(block=False))
            except Empty:
                break
        try:
            if ARGS.varlen:
                patches, sizes, cu_seq = image.varlen(
                    [work.take_image() for work in batch], 16,
                    max_n=ARGS.batch_size, device=device
                )
                outputs = MODEL.forward_varlen(MODEL.from_srgb(patches), sizes, cu_seq, seqlen)
                del patches, sizes, cu_seq
            else:
                patches, sizes = image.stack(
                    [work.take_image() for work in batch], 16, seqlen,
                    max_n=ARGS.batch_size, device=device
                )
                outputs = MODEL.forward(MODEL.from_srgb(patches), sizes)
                del patches, sizes
            WQ.queue(handle_batch, batch, outputs)
            del outputs
        except Exception as ex:
            for work in batch:
                work.error(ex)
        del batch

@APP.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root() -> Any:
    return "/docs"

@APP.get("/info",
    summary="Returns information about the model.",
    response_model=InfoResponse,
)
async def info() -> Any:
    bg = MODEL.image_config.background
    return {
        "model": MODEL.name,
        "extensions": EXTS,
        "image_config": {
            "colorspace": "srgb",
            "background": [bg, bg, bg] if not isinstance(bg, tuple) else list(bg),
            "resize_kernel": MODEL.image_config.resize_kernel,
            "resize_linear": MODEL.image_config.resize_linear,
            "patch_size": MODEL.image_config.patch_size,
            "max_seqlen": MODEL.image_config.max_seqlen,
        },
        "labels": {
            label.label: {
                "category": label.category,
                "subcategory": label.subcategory,
                "implies": label.implies,
            } for label in MODEL.labels
        },
    }

@APP.post("/classify",
    summary="Classifies the provided image.",
    response_model=ClassifyResponse,
    openapi_extra={
        "requestBody": {
            "description": "Image data to classify.",
            "required": True,
            "content": {
                "image/*": {
                    "schema": { "type": "string", "format": "binary" }
                }
            }
        }
    }
)

async def classify(
    request: Request,
    calibration: Annotated[str, Query(description=
        "Calibration metric. **Format:** *metric*[*arg*][*@min_prec*], or `none` to return all labels.\n\n"
        "- *metric*: `f`, `csi`\n"
        "- *arg*: Optional positive number trading off precision and recall, default `1.0`.\n"
        "- *min_prec*: Optional minimum precision floor, default `0.0`.\n"
    )] = "f1.6@0.0",
    implications: Annotated[str, Query(description=
        "Implications mode.\n\n"
        "- `preserve`           Tags are preserved if they are implied by another tag.\n"
        "- `inherit`            Tags are preserved and inherit the highest probability among the tags that imply them.\n"
        "- `constrain`          Tags are preserved and inherit the lowest probability among the tags they imply.\n"
        "- `enforce`            Tags are removed unless all the tags they imply are present.\n"
        "- `remove`             Exclude all implied tags.\n"
        "- `constrain-remove`   Combination of constrain followed by remove.\n"
        "- `enforce-inherit`    Combination of enforce followed by inherit.\n"
        "- `enforce-constrain`  Combination of enforce followed by constrain.\n"
        "- `enforce-remove`     Combination of enforce followed by remove.\n"
        "- `off`                Raw model output with no implications applied.",
        pattern=f"^(?:" + "|".join(IMPLICATION_MODES) + ")$"
    )] = "inherit",
):
    try:
        cal_obj = get_calibration(calibration)
    except Exception as ex:
        raise HTTPException(400, f"[{ex.__class__.__name__}] {ex}")
    data = await request.body()
    try:
        image = await to_thread(MODEL.load_image, data)
    except Exception as ex:
        raise HTTPException(415)
    del data
    future = get_running_loop().create_future()
    QUEUE.put(Work(
        image=image,
        calibration=cal_obj,
        implications=implications,
        future=future
    ))
    del image, cal_obj, implications
    return await future

class PredictRequest(BaseModel):
    data: list[Any]

@APP.middleware("http")
async def request_counter(request: Request, call_next):
    global REQUEST_COUNT
    response = await call_next(request)
    if request.url.path == "/run/predict":
        REQUEST_COUNT += 1
    if request.url.path == "/classify":
        REQUEST_COUNT += 1
    elapsed = int(time.time() - START_TIME)
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Requests Served: {REQUEST_COUNT:,}, "f"Uptime: {days} Days, {hours} Hours, {minutes} Minutes, {seconds} Seconds",end="\r",flush=True,)
    return response

@APP.post(
    "/run/predict",
    response_class=PlainTextResponse,
    include_in_schema=False,
)
async def predict(request: PredictRequest):
    if (
        len(request.data) < 2
        or not isinstance(request.data[0], str)
    ):
        raise HTTPException(400,'Expected {"data":["data:image/...;base64,...", threshold]}')
    image_uri = request.data[0]
    try:
        threshold = float(request.data[1])
    except Exception:
        raise HTTPException(400, "Invalid threshold")
    if not -1.0 <= threshold <= 1.0:
        raise HTTPException(400, "Threshold must be between -1.0 and 1.0")
    if not image_uri.startswith("data:"):
        raise HTTPException(400, "Invalid data URI")
    try:
        _, b64 = image_uri.split(",", 1)
        image_bytes = base64.b64decode(b64, validate=True)
    except Exception as ex:
        raise HTTPException(400, f"Invalid base64: {ex}")
    try:
        image = await to_thread(MODEL.load_image, image_bytes)
    except Exception as ex:
        raise HTTPException(415)
    future = get_running_loop().create_future()
    QUEUE.put(
        Work(
            image=image,
            calibration=MODEL.calibrate((threshold + 1.0) / 2.0),
            implications="inherit",
            future=future,
        )
    )
    result = await future
    tags = sorted(((tag, prob) for tag, prob in result.items() if tag not in EXCLUDED_TAGS), key=lambda x: (-x[1], x[0]))
    return " ".join(tag for tag, _ in tags)

def main() -> None:
    global MODEL, ARGS
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model  = "JTP-Hydra-3.5.safetensors"
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-i IP] [-p PORT] [-b BATCH_SIZE] [-d DEVICE] [...]",
        description="Hydra Classifier HTTP Service",
        allow_abbrev=False,
        add_help=False,
    )
    group = parser.add_argument_group("service")
    group.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )
    group.add_argument(
        "-i", "--ip", default="0.0.0.0",
        metavar="IP",
        help="Service IP address. (Default: 0.0.0.0)"
    )
    group.add_argument(
        "-p", "--port", type=int, default=22870,
        metavar="PORT",
        help="Service port. (Default: 22870)"
    )
    group = parser.add_argument_group("model")
    group.add_argument(
        "-M", "--model", default=default_model,
        metavar="PATH",
        help=f"Path to model file. (Default: {default_model})",
    )
    group.add_argument(
        "-D", "--metadata", default="./data",
        metavar="PATH",
        help="Metadata directory for legacy JTP-3 models. (Default: ./data)",
    )
    group.add_argument(
        "-e", "--extension", action="append", default=[],
        metavar="PATH",
        help=(
            "Path to extension. May be specified multiple times. "
            "If a directory is specified, all extensions in the specified directory are loaded. "
            "(Default: extensions/<model_name>)"
        ),
    )
    group.add_argument(
        "-E", "--no-default-extensions", action="store_true",
        help="Do not load extensions by default.",
    )
    group = parser.add_argument_group("execution")
    group.add_argument(
        "-b", "--batch-size", type=int, default=1,
        metavar="BATCH_SIZE",
        help="Maximum batch size. (Default: 1)"
    )
    group.add_argument(
        "-V", "--varlen", action="store_true",
        help="Use optimized varlen attention. (Requires flash attention support.)"
    )
    group.add_argument(
        "-S", "--seqlen", type=int, default=1024,
        help="NaFlex sequence length. (Default: 1024)"
    )
    group.add_argument(
        "-d", "--device", default=default_device,
        metavar="DEVICE",
        help=f"Torch device. (Default: {default_device})"
    )
    group.add_argument(
        "-c", "--compile", action="store_true",
        help="Compile the model for maximum performance."
    )
    ARGS = parser.parse_args()
    if ARGS.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 64 <= ARGS.seqlen <= 2048:
        parser.error("--seqlen must be between 64 and 2048")
    print(f"Loading {repr(ARGS.model)} ...", end="", file=sys.stderr)
    MODEL = load_model(ARGS.model, legacy_metadata_dir=ARGS.metadata)
    MODEL.image_config.max_seqlen = ARGS.seqlen
    print(f" {len(MODEL.labels)} tags.", file=sys.stderr)
    if not ARGS.extension and not ARGS.no_default_extensions:
        default_extensions = "extensions/" + os.path.splitext(os.path.basename(ARGS.model))[0]
        if os.path.isdir(default_extensions):
            ARGS.extension.append(default_extensions)
    if ARGS.extension:
        print(f"Loading extensions ...", file=sys.stderr)
        for ext in MODEL.load_extensions(Extension.discover(ARGS.extension)):
            EXTS.append(os.path.basename(ext.path))
            print(f"  {repr(ext.path)}: {repr(ext.label.label)} ({ext.label.category})", file=sys.stderr)
    if ARGS.device != "cpu":
        print(f"Transferring to device {repr(ARGS.device)}...", end="", file=sys.stderr)
        MODEL = MODEL.to(device=ARGS.device)
        print(f" done.", file=sys.stderr)
    if ARGS.compile:
        MODEL.compile(mode="max-autotune-no-cudagraphs")
    Thread(target=worker, name="device-worker", daemon=True).start()
    uvicorn.run(APP, host=ARGS.ip, port=ARGS.port, access_log=False)
if __name__ == "__main__":
    main()