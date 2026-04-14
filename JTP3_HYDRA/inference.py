import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterable, TypeAlias
import torch
from torch import Tensor
from loader import Loader
from model import discover_extensions, load_model, load_image
from siglip2 import NaFlexVit
from huggingface_hub import hf_hub_download
try:
    from itertools import batched
except ImportError:
    from itertools import islice
    def batched(iterable, n: int):
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch
Metadata: TypeAlias = dict[str, tuple[int, list[str]]]
Thresholds: TypeAlias = dict[str, float] | float
PATCH_SIZE = 16
TAG_CATEGORIES = {
    "general": 0,
    "artist": 1,
    "copyright": 3,
    "character": 4,
    "species": 5,
    "meta": 7,
    "lore": 8,
}
IMPLICATION_MODES = ("inherit", "constrain", "remove", "constrain-remove", "off")
EMPTY_METADATA: tuple[int, list[str]] = (-1, [])

print("Checking for model...")
if not os.path.isfile("./JTP-3-Hydra.safetensors"):
    print("Downloading tagger model, please wait...")
    path = hf_hub_download(repo_id="RedRocket/JTP-3", filename="models/jtp-3-hydra.safetensors", repo_type="model", local_dir="./")
    shutil.move(path, "./JTP-3-Hydra.safetensors")
    os.rmdir("models")
else:
    print("JTP-3-Hydra.safetensors found")

if not os.path.isfile("./JTP-3-Hydra-Tags.csv"):
    print("Downloading JTP-3-Hydra-Tags.csv")
    path = hf_hub_download(repo_id="RedRocket/JTP-3", filename="data/jtp-3-hydra-tags.csv", repo_type="model", local_dir="./")
    shutil.move(path, "./JTP-3-Hydra-Tags.csv")
    os.rmdir("data")
else:
    print("JTP-3-Hydra-Tags.csv found")
print("Model check complete!")

def from_symmetric(threshold: float) -> float:
    return (threshold + 1.0) / 2.0

def to_symmetric(threshold: float) -> float:
    return (threshold - 0.5) * 2.0

def inherit_implications(
    labels: dict[str, float], antecedent: str,
    metadata: Metadata,
) -> None:
    p = labels[antecedent]
    for consequent in metadata.get(antecedent, EMPTY_METADATA)[1]:
        if (q := labels.get(consequent)) is None:
            continue
        if q < p:
            labels[consequent] = p
        inherit_implications(labels, consequent, metadata)

def constrain_implications(
    labels: dict[str, float], antecedent: str,
    metadata: Metadata,
    *, _target: str | None = None
) -> None:
    if _target is None:
        _target = antecedent
    for consequent in metadata.get(antecedent, EMPTY_METADATA)[1]:
        if (p := labels.get(consequent)) is None:
            continue
        if labels[_target] > p:
            labels[_target] = p
        constrain_implications(labels, consequent, metadata, _target=_target)

def remove_implications(
    labels: dict[str, float], antecedent: str,
    metadata: Metadata,
) -> None:
    for consequent in metadata.get(antecedent, EMPTY_METADATA)[1]:
        labels.pop(consequent, None)
        remove_implications(labels, consequent, metadata)

def classify_output(
    output: Tensor,
    tags: list[str],
    threshold: Thresholds = 0.0,
    *,
    metadata: Metadata = {},
    implications: str = "off",
    exclude_categories: set[int] | frozenset[int] = frozenset(),
) -> dict[str, float]:
    labels = dict(zip(tags, output.tolist(), strict=True))
    match implications:
        case "inherit":
            for tag in tags:
                inherit_implications(labels, tag, metadata)
        case "constrain" | "constrain-remove":
            for tag in tags:
                constrain_implications(labels, tag, metadata)
        case "remove" | "off":
            pass
        case _:
            raise ValueError("Invalid implications mode.")
    labels = {
        tag: prob
        for tag, prob in labels.items()
        if (
            not exclude_categories
            or metadata.get(tag, EMPTY_METADATA)[0] not in exclude_categories
        ) and prob >= (
            threshold.get(tag, float("inf"))
            if isinstance(threshold, dict)
            else threshold
        )
    }
    if implications in ("remove", "constrain-remove"):
        for tag in tags:
            if tag in labels:
                remove_implications(labels, tag, metadata)
    return labels

def _run_interactive(
    *,
    model: NaFlexVit,
    tags: list[str],
    threshold: Thresholds,
    metadata: Metadata,
    implications: str,
    exclude: set[int],
    seqlen: int,
    device: str,
    rewrite_tag: Callable[[str], str],
) -> None:
    print(
        "\n"
        "JTP-3 Hydra Interactive Classifier\n"
        "  Type 'q' to quit, or 'h' for help.\n"
        "  For bulk operations, quit and run again with a path, or '-h' for help.\n"
    )
    while True:
        print("> ", end="")
        line = input().strip()
        if line in ("q", "quit", "exit"):
            break
        if line in ("", "h", "help", "?"):
            print(
                "Provide a file path to classify, or one of the following commands:\n"
                "  threshold NUM      (-1.0 to 1.0, 0.2 to 0.8 recommended)\n"
                "  calibration [PATH] (load calibration csv file)"
            )
            if metadata:
                print(
                    f"  exclude CATEGORY   ({' '.join(TAG_CATEGORIES.keys())})\n"
                    f"  include CATEGORY   ({' '.join(TAG_CATEGORIES.keys())})\n"
                    f"  implications MODE  ({' '.join(IMPLICATION_MODES)})"
                )
            print(
                "  seqlen LEN         (64 to 2048, 1024 recommended)\n"
                "  quit               (or 'q', 'exit')"
            )
            continue
        if line.startswith("threshold "):
            try:
                parsed = float(line[10:])
            except Exception as ex:
                print(ex)
                continue
            if -1.0 <= parsed <= 1.0:
                threshold = from_symmetric(parsed)
            else:
                print("Threshold must be between -1.0 and 1.0.")
            continue
        if line == "calibration":
            try:
                threshold = load_calibration("calibration.csv", rewrite_tag)
            except Exception as ex:
                print(ex)
            continue
        if line.startswith("calibration "):
            try:
                threshold = load_calibration(line[12:], rewrite_tag)
            except Exception as ex:
                print(ex)
            continue
        if line.startswith("seqlen "):
            try:
                parsed = int(line[7:])
            except Exception as ex:
                print(ex)
                continue
            if 64 <= parsed <= 2048:
                seqlen = parsed
            else:
                print("Sequence length must be between 64 and 2048.")
            continue
        if line.startswith("exclude "):
            if not metadata:
                print("Tag metadata is not loaded.")
                continue
            try:
                exclude.add(TAG_CATEGORIES[line[8:]])
            except KeyError:
                print(f"Category must be one of: {' '.join(TAG_CATEGORIES.keys())}")
            continue
        if line.startswith("include "):
            try:
                exclude.discard(TAG_CATEGORIES[line[8:]])
            except KeyError:
                print(f"Category must be one of: {' '.join(TAG_CATEGORIES.keys())}")
            continue
        if line.startswith("implications "):
            if not metadata and line[13:] != "off":
                print("Tag metadata is not loaded.")
                continue
            if line[13:] in IMPLICATION_MODES:
                implications = line[13:]
            else:
                print(f"Mode must be one of: {' '.join(IMPLICATION_MODES)}")
            continue
        try:
            p_t, pc_t, pv_t = load_image(line, PATCH_SIZE, seqlen, False)
        except Exception as ex:
            print(ex)
            continue
        p_d = p_t.unsqueeze(0).to(device=device, non_blocking=True)
        pc_d = pc_t.unsqueeze(0).to(device=device, non_blocking=True)
        pv_d = pv_t.unsqueeze(0).to(device=device, non_blocking=True)
        p_d = p_d.to(dtype=torch.bfloat16).div_(127.5).sub_(1.0)
        pc_d = pc_d.to(dtype=torch.int32)
        o_d = model(p_d, pc_d, pv_d).float().sigmoid()
        del p_d, pc_d, pv_d
        classes = classify_output(
            o_d[0], tags, threshold,
            metadata=metadata,
            implications=implications,
            exclude_categories=exclude,
        )
        for cls, prob in sorted(classes.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {to_symmetric(prob)*100:6.1f}% {cls}")
        del classes
        del o_d
        del p_t, pc_t, pv_t

def _run_batched(
    *,
    model: NaFlexVit,
    tags: list[str],
    paths: list[str],
    recursive: bool,
    metadata: dict[str, tuple[int, list[str]]],
    implications: str,
    exclude: set[int],
    threshold: dict[str, float] | float,
    writer: Any,
    prefix: str,
    batch_size: int,
    seqlen: int,
    n_workers: int,
    share_memory: bool,
    device: str,
) -> None:
    loader = Loader(
        n_workers,
        patch_size=PATCH_SIZE, max_seqlen=seqlen,
        share_memory=share_memory
    )

    def dir_iter(path: str) -> Iterable[str]:
        for entry in os.scandir(path):
            if (
                not entry.name.startswith(".")
                and entry.name != "__pycache__"
            ):
                if entry.is_file():
                    if not entry.name.endswith((
                        ".txt", ".csv", ".json",
                        ".py", ".safetensors",
                    )):
                        yield entry.path
                elif recursive and entry.is_dir():
                    yield from dir_iter(entry.path)

    def paths_iter() -> Iterable[str]:
        for path in paths:
            if os.path.isdir(path):
                yield from dir_iter(path)
            else:
                yield path
    for batch in batched(paths_iter(), batch_size):
        patches: list[Tensor] = []
        patch_coords: list[Tensor] = []
        patch_valid: list[Tensor] = []
        batch_paths: list[str] = []
        for path, result in loader.load(batch).items():
            if isinstance(result, Exception):
                print(f"{repr(path)}: {result}", file=sys.stderr)
                continue
            batch_paths.append(path)
            patches.append(result[0])
            patch_coords.append(result[1])
            patch_valid.append(result[2])
        if not patches:
            continue
        p_d = torch.stack(patches).to(device=device, non_blocking=True)
        pc_d = torch.stack(patch_coords).to(device=device, non_blocking=True)
        pv_d = torch.stack(patch_valid).to(device=device, non_blocking=True)
        p_d = p_d.to(dtype=torch.bfloat16).div_(127.5).sub_(1.0)
        pc_d = pc_d.to(dtype=torch.int32)
        o_d = model(p_d, pc_d, pv_d).float().sigmoid()
        del p_d, pc_d, pv_d
        for path, output in zip(batch_paths, o_d.cpu()):
            if writer is None:
                with open(
                    f"{os.path.splitext(path)[0]}.txt", "w",
                    encoding="utf-8"
                ) as file:
                    classes = list(classify_output(
                        output, tags, threshold,
                        metadata=metadata, implications=implications, exclude_categories=exclude
                    ).keys())
                    random.shuffle(classes)
                    if prefix:
                        try:
                            classes.remove(prefix)
                        except ValueError:
                            pass
                        classes.insert(0, prefix)
                    file.write(', '.join(classes))
            else:
                writer.writerow((path, *(f"{prob.item():.4f}" for prob in output)))
        del o_d
    loader.shutdown()

def load_calibration(path: str, rewrite_tag: Callable[[str], str] = lambda tag: tag) -> dict[str, float]:
    thresholds = {}
    with open(path, "r", encoding="utf-8", newline="") as thresholds_file:
        reader = csv.DictReader(thresholds_file)
        if (
            "tag" not in reader.fieldnames
            or "threshold" not in reader.fieldnames
        ):
            raise RuntimeError("CSV must have the columns 'tag' and 'threshold'")
        for row in reader:
            if not row["threshold"]:
                continue
            try:
                value = float(row["threshold"])
            except ValueError as ex:
                raise RuntimeError("'threshold' must be between 0.0 and 1.0, or blank") from ex
            if not 0.0 <= value <= 1.0:
                raise RuntimeError("'threshold' must be between 0.0 and 1.0, or blank")
            thresholds[rewrite_tag(row["tag"])] = value
    return thresholds

def load_metadata(path: str, rewrite_tag: Callable[[str], str] = lambda tag: tag) -> dict[str, tuple[int, list[str]]]:
    metadata = {}
    with open(path, "r", encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if (
            "tag" not in reader.fieldnames
            or "category" not in reader.fieldnames
            or "implications" not in reader.fieldnames
        ):
            raise RuntimeError("CSV must have the columns 'tag', 'category', and 'implications'")
        for row in reader:
            metadata[rewrite_tag(row["tag"])] = (int(row["category"]), [
                rewrite_tag(tag)
                for tag in row["implications"].split()
            ])
    return metadata

def _if_exists(path: str, default: str = "") -> str:
    return path if os.path.exists(path) else default

def _run_service(
    *,
    model: NaFlexVit, tags: list[str],
    seqlen: int, threshold: float,
    device: str, host: str, port: int
) -> None:
    """Run HTTP service for image classification via base64 JSON API."""
    try:
        from flask import Flask, request
        import base64
        import mimetypes
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

    except ImportError:
        print("Error: Flask is required for --service mode. Install it with: pip install flask", file=sys.stderr)
        sys.exit(1)

    app = Flask(__name__)
    request_count = 0
    import time
    start_time = time.time()
    def parse_data_uri(data_uri: str) -> tuple[str, str]:
        """Parse data URI and return (mime_type, base64_data)"""
        if not data_uri.startswith('data:'):
            raise ValueError("Invalid data URI format")
        header, data = data_uri.split(',', 1)
        mime_part = header.split(';')[0]
        if ':' in mime_part:
            mime_type = mime_part.split(':', 1)[1]
        else:
            mime_type = 'application/octet-stream'
        return mime_type, data

    @app.route('/run/predict', methods=['POST'])
    def classify_image():
        nonlocal request_count
        request_count += 1
        elapsed = int(time.time() - start_time)
        days = elapsed // 86400
        hours = (elapsed % 86400) // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        print(f"Requests Served: {request_count}, Uptime: {days} Days, {hours} Hours, {minutes} Minutes, {seconds} Seconds", end="\r", flush=True)
        """Accept base64 image via JSON POST and return space-separated tags."""
        if not request.is_json:
            return "Content-Type must be application/json", 400
        try:
            data = request.get_json()
            if 'data' not in data or not isinstance(data['data'], list) or len(data['data']) < 2:
                return "Invalid JSON format. Expected: {\"data\": [\"data:image/...;base64,...\", threshold]}", 400
            image_uri = data['data'][0]
            request_threshold = float(data['data'][1])
            if not -1.0 <= request_threshold <= 1.0:
                return f"Threshold must be between -1.0 and 1.0, got {request_threshold}", 400
            mime_type, base64_data = parse_data_uri(image_uri)
            try:
                image_data = base64.b64decode(base64_data, validate=True)
            except Exception as e:
                return f"Invalid base64 encoding: {e}", 400
            ext = mimetypes.guess_extension(mime_type)
            if not ext:
                if mime_type.startswith('image/'):
                    ext = '.' + mime_type.split('/')[-1]
                else:
                    ext = '.bin'
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(image_data)
                    tmp_path = tmp.name
                with torch.inference_mode():
                    p_t, pc_t, pv_t = load_image(tmp_path, PATCH_SIZE, seqlen, False)
                    p_d = p_t.unsqueeze(0).to(device=device, non_blocking=True)
                    pc_d = pc_t.unsqueeze(0).to(device=device, non_blocking=True)
                    pv_d = pv_t.unsqueeze(0).to(device=device, non_blocking=True)
                    p_d = p_d.to(dtype=torch.bfloat16).div_(127.5).sub_(1.0)
                    pc_d = pc_d.to(dtype=torch.int32)
                    o_d = model(p_d, pc_d, pv_d).float().sigmoid()
                    converted_threshold = from_symmetric(request_threshold)
                    classes_dict = classify_output(o_d[0], tags, converted_threshold)
                    sorted_tags = [tag for tag, prob in sorted(classes_dict.items(), key=lambda item: (-item[1], item[0]))]
                    del p_d, pc_d, pv_d, o_d, p_t, pc_t, pv_t
                    response_text = ' '.join(sorted_tags)
                    return response_text, 200, {'Content-Type': 'text/plain'}
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        except Exception as ex:
            print(ex)
            return f"Error processing request: {ex}", 500

    @app.route('/', methods=['GET'])
    def index():
        """Service information endpoint."""
        return "JTP³ Hydra Image Classification Service\nPOST JSON to /run/predict with format: {\"data\": [\"data:image/...;base64,...\", confidence]}", 200, {'Content-Type': 'text/plain'}
    print(f"Starting HTTP service on {host}:{port}", file=sys.stderr)
    print(f"POST base64 images to http://{host}:{port}/run/predict", file=sys.stderr)
    print("Press Ctrl+C to stop", file=sys.stderr)
    try:
        from waitress import serve
        serve(app, host=host, port=port)
    except ImportError:
        app.run(host=host, port=port, threaded=False)

@torch.inference_mode()
def main() -> None:
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_threshold = _if_exists("calibration.csv", "0.5")
    default_metadata = _if_exists("JTP-3-Hydra-Tags.csv")
    default_extension = _if_exists("jtp-3-hydra")
    parser = argparse.ArgumentParser(
        description="JTP-3 Hydra",
        epilog="By Project RedRocket. Visit https://huggingface.co/spaces/RedRocket/JTP-3 for more information."
    )
    group = parser.add_argument_group("classification")
    group.add_argument(
        "-t", "--threshold", default=default_threshold,
        metavar="THRESHOLD_OR_PATH",
        help=f"Classification threshold -1.0 to 1.0. Or, a path to a CSV calibration file. (Default: {default_threshold})"
    )
    group.add_argument(
        "-i", "--implications", choices=IMPLICATION_MODES,
        metavar="MODE",
        help="Automatically apply implications. Requires tag metadata. (Default: inherit)"
    )
    group.add_argument(
        "-x", "--exclude", action="append", choices=TAG_CATEGORIES.keys(), default=[],
        metavar="CATEGORY",
        help="Exclude the specified category of tags. May be specified multiple times. Requires tag metadata."
    )
    # OUTPUT ARGUMENTS
    group = parser.add_argument_group("output")
    group.add_argument(
        "-p", "--prefix", default="",
        help="Prefix all .txt caption files with the specified text. If the prefix matches a tag, the tag will not be repeated."
    )
    group.add_argument(
        "-o", "--output",
        metavar="PATH",
        help="Path for CSV output, or '-' for standard output. If not specified, individual .txt caption files are written."
    )
    group.add_argument(
        "-O", "--original-tags", action="store_true",
        help="Do not rewrite tags for compatibility with diffusion models."
    )
    group = parser.add_argument_group("model")
    group.add_argument(
        "-M", "--model", default="JTP-3-Hydra.safetensors",
        metavar="PATH",
        help="Path to model file."
    )
    group.add_argument(
        "-m", "--metadata", default=default_metadata,
        metavar="PATH",
        help=f"Path to CSV file with additional tag metadata. (Default: {default_metadata or '<none>'})"
    )
    group.add_argument(
        "-e", "--extension", action="append", default=[],
        metavar="PATH",
        help=(
            "Path to extension. May be specified multiple times. "
            "If a directory is specified, all extensions in the specified directory are loaded. "
            f"(Default: {default_extension or '<none>'})"
        )
    )
    group.add_argument(
        "-E", "--no-default-extensions", action="store_true",
        help="Do not load extensions by default."
    )
    group = parser.add_argument_group("execution")
    group.add_argument(
        "-b", "--batch", type=int, default=1,
        metavar="BATCH_SIZE",
        help="Batch size."
    )
    group.add_argument(
        "-w", "--workers", type=int, default=-1,
        metavar="N_WORKERS",
        help="Number of dataloader workers. (Default: number of cores)"
    )
    group.add_argument(
        "--no-shm", dest="shm", action="store_false",
        help="Disable shared memory between workers."
    )
    group.add_argument(
        "-S", "--seqlen", type=int, default=1024,
        help="NaFlex sequence length. (Default: 1024)"
    )
    group.add_argument(
        "-d", "--device", default=default_device,
        metavar="TORCH_DEVICE",
        help=f"Torch device. (Default: {default_device})"
    )
    # SERVER ARGUMENTS
    parser.add_argument("--service", action="store_true",
        help="Run as HTTP service. Enables /image endpoint for POST requests. Cannot be used with paths or other batch options.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
        help="Host to bind HTTP service to. (Default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=22870,
        help="Port to bind HTTP service to. (Default: 22870)")
    # INPUT ARGUMENTS
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Classify directories recursively. Dotfiles will be ignored."
    )
    parser.add_argument(
        "paths", nargs="*",
        metavar="PATH",
        help="Paths to files and directories to classify. If none are specified, run interactively."
    )
    args = parser.parse_args()
    def rewrite_tag(tag: str) -> str:
        if args.service:
            tag = tag.replace("safe", "")
            tag = tag.replace("mature", "")
            tag = tag.replace("questionable", "")
            tag = tag.replace("explicit", "")
        if not args.original_tags:
            tag = tag.replace("vulva", "pussy")
        if args.output is None and args.paths: # caption files
            tag = tag.replace("_", " ")
            tag = tag.replace("(", r"\(")
            tag = tag.replace(")", r"\)")
        return tag
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    if not 64 <= args.seqlen <= 2048:
        parser.error("--seqlen must be between 64 and 2048")
    threshold: dict[str, float] | float
    try:
        threshold = float(args.threshold)
        if not -1.0 <= threshold <= 1.0:
            parser.error("--threshold value must be between -1.0 and 1.0")
        threshold = from_symmetric(threshold)
    except ValueError: # not a float, try to interpret as path to a calibration file
        print(f"Loading {repr(args.threshold)} ...", end="", file=sys.stderr)
        threshold = load_calibration(args.threshold, rewrite_tag)
        print(f" {len(threshold)} tags", file=sys.stderr)
    metadata: Metadata = {}
    if args.metadata is not None:
        print(f"Loading {repr(args.metadata)} ...", end="", file=sys.stderr)
        metadata = load_metadata(args.metadata, rewrite_tag)
        print(f" {len(metadata)} tags", file=sys.stderr)
    if args.implications is None:
        args.implications = "inherit" if metadata else "off"
    elif args.implications != "off" and not metadata:
        parser.error(f"--implications {args.implications} requires tag metadata")
    if args.exclude and not metadata:
        parser.error("--exclude requires tag metadata")
    if (
        not args.extension
        and not args.no_default_extensions
        and default_extension
    ):
        args.extension.append(default_extension)
    print(f"Loading {repr(args.model)} ...", end="", file=sys.stderr)
    model, tags, ext_info = load_model(
        args.model,
        extensions=discover_extensions(args.extension),
        device=args.device
    )
    print(f" {len(tags)} tags", file=sys.stderr)
    for idx in range(len(tags)):
        tags[idx] = rewrite_tag(tags[idx])
    for ext_path, info in ext_info.items():
        tag = rewrite_tag(info['label'])
        if not isinstance(threshold, dict) or tag in threshold:
            print(f"Using tag {repr(tag)} ({info['category']}) from extension {repr(ext_path)}.", file=sys.stderr)
        else:
            print(f"Ignoring tag {repr(tag)} from uncalibrated extension {repr(ext_path)}.", file=sys.stderr)
    if args.metadata is not None:
        for info in ext_info.values():
            metadata[info["label"]] = (
                TAG_CATEGORIES.get(info["category"], -1),
                [rewrite_tag(impl) for impl in info["implies"].split()]
            )
    exclude = { TAG_CATEGORIES[category] for category in args.exclude }
    if args.paths:
        file: Any = None
        writer: Any = None
        match args.output:
            case None:
                pass
            case "-":
                writer = csv.writer(sys.stdout)
            case _:
                file = open(
                    args.file, "w",
                    buffering=(1024 * 1024),
                    encoding="utf-8",
                    newline="",
                )
                writer = csv.writer(file)
                writer.writerow(("filename", *tags))
        try:
            _run_batched(
                model=model, tags=tags,
                threshold=threshold,
                metadata=metadata, implications=args.implications, exclude=exclude,
                paths=args.paths, recursive=args.recursive,
                writer=writer, prefix=args.prefix,
                batch_size=args.batch, seqlen=args.seqlen,
                n_workers=args.workers, share_memory=args.shm,
                device=args.device,
            )
        finally:
            if file is not None:
                file.close()
    elif args.service:
        _run_service(
            model=model,
            tags=tags,
            threshold=args.threshold,
            seqlen=args.seqlen,
            device=args.device,
            host=args.host,
            port=args.port,
        )
    else:
        _run_interactive(
            model=model, tags=tags, rewrite_tag=rewrite_tag,
            threshold=threshold,
            metadata=metadata, implications=args.implications, exclude=exclude,
            seqlen=args.seqlen,
            device=args.device,
        )

if __name__ == "__main__":
    main()