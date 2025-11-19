import argparse
import csv
import itertools
import os
import random
import shutil
import sys
import tempfile
from typing import Any, Iterable
import torch
from torch import Tensor
from timm.models import NaFlexVit
from loader import Loader
from model import load_model, load_image
from huggingface_hub import hf_hub_download
PATCH_SIZE = 16

print("Checking for model...")
if not os.path.isfile("./jtp-3-hydra.safetensors"):
    print("Downloading tagger model, please wait...")
    path = hf_hub_download(repo_id="RedRocket/JTP-3", filename="models/jtp-3-hydra.safetensors", repo_type="model", local_dir="./")
    shutil.move(path, "./jtp-3-hydra.safetensors")
    os.rmdir("models")
else:
    print("jtp-3-hydra.safetensors found")

if not os.path.isfile("./jtp-3-hydra-tags.csv"):
    print("Downloading jtp-3-hydra-tags.csv")
    path = hf_hub_download(repo_id="RedRocket/JTP-3", filename="data/jtp-3-hydra-tags.csv", repo_type="model", local_dir="./")
    shutil.move(path, "./jtp-3-hydra-tags.csv")
    os.rmdir("data")
else:
    print("jtp-3-hydra-tags.csv found")
print("Model check complete!")

def from_symmetric(threshold: float) -> float:
    return (threshold + 1.0) / 2.0

def to_symmetric(threshold: float) -> float:
    return (threshold - 0.5) * 2.0

def classify_output(output: Tensor, tags: list[str], threshold: float = 0.0) -> dict[str, float]:
    return {
        tag: prob
        for tag, prob in zip(tags, output.tolist())
        if prob >= threshold
    }

def _run_interactive(
    *,
    model: NaFlexVit, tags: list[str],
    seqlen: int, threshold: float,
    device: str
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
                f"  threshold T   (-1.0 to 1.0, currently {threshold}, 0.2 to 0.8 recommended)\n"
                f"  seqlen N      (64 to 2048, currently {seqlen}, 1024 recommended)\n"
                "  quit          (or 'q', 'exit')"
            )
            continue
        if line.startswith("threshold "):
            try:
                parsed = float(line[10:])
            except Exception as ex:
                print(ex)
                continue
            if -1.0 <= parsed <= 1.0:
                threshold = parsed
            else:
                print("Threshold must be between -1.0 and 1.0.")
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
        classes = classify_output(o_d[0], tags, from_symmetric(threshold))
        for cls, prob in sorted(classes.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {to_symmetric(prob)*100:6.1f}% {cls}")
        del classes
        del o_d
        del p_t, pc_t, pv_t

def _run_batched(
    *,
    model: NaFlexVit, tags: list[str],
    paths: list[str], recursive: bool,
    threshold: float, writer: Any, prefix: str,
    batch_size: int, seqlen: int,
    n_workers: int, share_memory: bool,
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
    for batch in itertools.batched(paths_iter(), batch_size):
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
                    classes = list(classify_output(output, tags, threshold).keys())
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="JTP-3 Hydra",
        epilog="By Project RedRocket. Visit https://huggingface.co/spaces/RedRocket/JTP-3 for more information."
    )
    parser.add_argument("--model", type=str, default="jtp-3-hydra.safetensors",
        help="Path to model file.")
    parser.add_argument("-b", "--batch", type=int, default=1,
        help="Batch size.")
    parser.add_argument("-w", "--workers", type=int, default=-1,
        help="Number of dataloader workers. (Default: number of cores)")
    parser.add_argument("--seqlen", type=int, default=1024,
        help="NaFlex sequence length. (Default: 1024)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
        help="Classification threshold. (-1.0 to 1.0)")
    parser.add_argument("--no-shm", dest="shm", action="store_false",
        help="Disable shared memory between workers.")
    parser.add_argument("-d", "--device", type=str, default=default_device,
        help=f"Torch device. (Default: {default_device})")
    parser.add_argument("-r", "--recursive", action="store_true",
        help="Classify directories recursively. (Dotfiles will be ignored.)")
    parser.add_argument("-O", "--original-tags", action="store_true",
        help="Do not rewrite tags for compatibility with diffusion models.")
    parser.add_argument("-o", "--output", type=str,
        help="Path for CSV output, or '-' for standard output. If not specified, individual .txt caption files are written.")
    parser.add_argument("-p", "--prefix", type=str, default="",
        help="Prefix all .txt caption files with the specified text. If the prefix matches a tag, the tag will not be repeated.")
    parser.add_argument("--service", action="store_true",
        help="Run as HTTP service. Enables /image endpoint for POST requests. Cannot be used with paths or other batch options.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
        help="Host to bind HTTP service to. (Default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=22870,
        help="Port to bind HTTP service to. (Default: 22870)")
    parser.add_argument("paths", nargs="*",
        help="Path to files and directories to classify. If none are specified, run interactively.")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    if not 64 <= args.seqlen <= 2048:
        parser.error("--seqlen must be between 64 and 2048 (1024 strongly recommended)")
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between -1.0 and 1.0")
    print(f"Loading {repr(args.model)} ...", end="", file=sys.stderr)
    model, tags = load_model(args.model, device=args.device)
    print(f" {len(tags)} tags", file=sys.stderr)

    def rewrite_tag(tag: str) -> str:
        if args.service:
            return tag
        if not args.original_tags:
            tag = tag.replace("vulva", "pussy")
        if args.output is None and args.paths: # caption files
            tag = tag.replace("_", " ")
            tag = tag.replace("(", r"\(")
            tag = tag.replace(")", r"\)")
        return tag
    for idx in range(len(tags)):
        tags[idx] = rewrite_tag(tags[idx])
    if args.service:
        _run_service(
            model=model,
            tags=tags,
            seqlen=args.seqlen,
            threshold=args.threshold,
            device=args.device,
            host=args.host,
            port=args.port,
        )
    elif args.paths:
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
                model=model,
                tags=tags,
                paths=args.paths,
                recursive=args.recursive,
                threshold=from_symmetric(args.threshold),
                writer=writer, prefix=args.prefix,
                batch_size=args.batch,
                seqlen=args.seqlen,
                n_workers=args.workers,
                share_memory=args.shm,
                device=args.device,
            )
        finally:
            if file is not None:
                file.close()
    else:
        _run_interactive(
            model=model,
            tags=tags,
            seqlen=args.seqlen,
            threshold=args.threshold,
            device=args.device,
        )
if __name__ == "__main__":
    main()
