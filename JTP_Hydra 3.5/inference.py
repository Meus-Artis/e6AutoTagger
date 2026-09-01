import argparse
import csv
import os
import sys
import time

from itertools import chain
from math import ceil
from threading import Event
from traceback import print_exc
from typing import Any, Callable, Iterable, Sequence, TextIO

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import logging
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import torch
from torch import Tensor

from hydra import image
from hydra.label import Label, Rewriter, parse_list, parse_aliases, TAG_CATEGORIES
from hydra.classification import Calibration, ExclusiveGroup, IMPLICATION_MODES
from hydra.model import Hydra, Extension, load_model

from utils.loader import Loader
from utils.workqueue import WorkQueue

def _dir_iter(path: str, recursive: bool) -> Iterable[str]:
    for entry in os.scandir(path):
        if entry.name.startswith("."):
            continue

        if entry.is_file():
            if entry.name.lower().endswith((
                ".avif", ".bmp", ".gif", ".jpeg", ".jpg",
                ".jxl", ".png", ".tiff", ".webp",
            )):
                yield entry.path
        elif recursive and entry.is_dir() and not entry.name.startswith("__"):
            yield from _dir_iter(entry.path, True)

def paths_iter(paths: Iterable[str], recursive: bool) -> Iterable[str]:
    for path in paths:
        if os.path.isdir(path):
            yield from _dir_iter(path, recursive)
        else:
            yield path

@torch.inference_mode()
def process_batched(
    model: Hydra,
    loader: Loader,
    paths: Iterable[str],
    *,
    batch_size: int,
    calibration: Calibration | None = None,
    csv_output: TextIO | None = None,
    implications: str = "inherit",
    exclude_categories: set[str] | frozenset[str] | str = frozenset(),
    exclude_tags: set[str] | frozenset[str] | str = frozenset(),
    exclusive_groups: Sequence[ExclusiveGroup] | str = (),
    rewrite: dict[str, Any] = {},
    varlen_attn: bool = False,
    workqueue: WorkQueue | None = None,
    on_done: Callable[[str, dict[str, float]], None] = lambda _a, _b: None,
    on_error: Callable[[str, Exception], None] = lambda _a, _b: None,
    stop_event: Event | None = None,
) -> None:
    if loader:
        raise RuntimeError("Loader is busy.")

    loader.queue_from(paths)

    if csv_output is None:
        if calibration is None:
            raise ValueError("Non-CSV output requires a calibration.")

        if isinstance(exclude_categories, str):
            exclude_categories = set(exclude_categories.split())

        if isinstance(exclude_tags, str):
            exclude_tags = set(parse_list(exclude_tags))

        if isinstance(exclusive_groups, str):
            exclusive_groups = list(ExclusiveGroup.parse_file(exclusive_groups))

        rewriter: Rewriter
        if (rewriter := rewrite.get("rewriter")) is None: # type: ignore[assignment]
            rewriter = Rewriter.create(
                model.labels,
                aliases=rewrite.get("aliases", {}),
                prefixes=rewrite.get("prefixes", {}),
                spaces=rewrite.get("spaces", False),
                escape=rewrite.get("escape", False),
            )
            rewriter.check()

        @torch.inference_mode()
        def handler_fn(paths: list[str], outputs: Tensor) -> None:
            outputs = outputs.cpu()
            for path, output in zip(paths, outputs.unbind(0)):
                try:
                    labels = calibration.classify_output(
                        output,
                        implications=implications,
                        exclude_categories=exclude_categories,
                        exclude_labels=exclude_tags,
                        exclusive_groups=exclusive_groups,
                        sort=False,
                    )

                    with open(
                        f"{os.path.splitext(path)[0]}.txt", "w",
                        encoding="utf-8",
                    ) as file:
                        file.write(rewriter.rewrite_join(
                            labels.keys(),
                            prefix=rewrite.get("prefix"),
                            shuffle=rewrite.get("shuffle", False),
                        ))
                except Exception as ex:
                    on_error(path, ex)
                else:
                    on_done(path, labels)
    else:
        writer = csv.writer(csv_output)
        writer.writerow(chain(("filename",), model.label_names()))

        @torch.inference_mode()
        def handler_fn(paths: list[str], outputs: Tensor) -> None:
            outputs = outputs.cpu()
            for path, output in zip(paths, outputs.unbind(0)):
                try:
                    writer.writerow(chain((path,), (f"{prob:.4f}" for prob in output.tolist())))
                except Exception as ex:
                    on_error(path, ex)
                else:
                    on_done(path, {})

    device = model.embeds.pos_embed.device
    seqlen = model.image_config.max_seqlen

    if workqueue is None:
        workqueue = WorkQueue(name="cpu-worker")
        owns_wq = True
    else:
        owns_wq = False

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            batch, errors = loader.get_batch(batch_size)

            if stop_event is not None and stop_event.is_set():
                break

            for path, ex in errors:
                on_error(path, ex)

            if not batch:
                break

            batch_paths: list[str] = []
            images: list[Tensor] = []
            for path, img in batch:
                batch_paths.append(path)
                images.append(img)

            del batch

            if varlen_attn:
                patches, sizes, cu_seq = image.varlen(images, 16, device=device)
                del images

                outputs = model.forward_varlen(
                    model.from_srgb(patches),
                    sizes, cu_seq, seqlen,
                )
                del patches, sizes, cu_seq
            else:
                patches, sizes = image.stack(images, 16, seqlen, device=device)
                del images

                outputs = model.forward(model.from_srgb(patches), sizes)
                del patches, sizes

            workqueue.queue(handler_fn, batch_paths, outputs)
            del outputs
    finally:
        if owns_wq:
            workqueue.shutdown()
        else:
            workqueue.wait()

    if stop_event is not None and stop_event.is_set():
        loader.clear()

@torch.inference_mode()
def main() -> None:
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model  = "models/JTP-Hydra-3.5.safetensors"
    default_workers = min(int(ceil((os.cpu_count() or 1) / 3)), 16)

    parser = argparse.ArgumentParser(
        usage="%(prog)s [-b BATCH_SIZE] [-w N_WORKERS] [-d DEVICE] [-o CSV] [-r] [...] [--] PATH...",
        description="Hydra Classifier by Project RedRocket",
        epilog=(
            "METRIC: metric[arg][@min_prec]\n"
            "  metric: f, csi\n"
            "  arg: Optional positive number trading off precision and recall, default 1.0.\n"
            "  min_prec: Optional minimum precision floor, default 0.0.\n"
            "\n"
            "MODE:\n"
            "  preserve           Tags are preserved if they are implied by another tag.\n"
            "  inherit            Tags are preserved and inherit the highest probability among the tags that imply them.\n"
            "  constrain          Tags are preserved and inherit the lowest probability among the tags they imply.\n"
            "  enforce            Tags are removed unless all the tags they imply are present.\n"
            "  remove             Exclude all implied tags.\n"
            "  constrain-remove   Combination of constrain followed by remove.\n"
            "  enforce-inherit    Combination of enforce followed by inherit.\n"
            "  enforce-constrain  Combination of enforce followed by constrain.\n"
            "  enforce-remove     Combination of enforce followed by remove.\n"
            "  off                Raw model output with no implications applied.\n"
            "\n"
            "CATEGORY:\n"
            f"  {' '.join(TAG_CATEGORIES)} color rating\n"
            f"  (or other categories supported by the model or extensions)\n"
            "\n"
            "Visit https://huggingface.co/RedRocket/Hydra for more information."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        add_help=False,
    )

    group = parser.add_argument_group("input")
    group.add_argument(
        "-r", "--recursive", action="store_true",
        help="Classify directories recursively. Dotfiles will be ignored.",
    )
    group.add_argument(
        "paths", nargs="+",
        metavar="PATH",
        help="Paths to files and directories to classify.",
    )

    group = parser.add_argument_group("output")
    group.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )
    group.add_argument(
        "-o", "--output",
        metavar="PATH",
        help=(
            "Path for CSV output, or '-' for standard output. "
            "If not specified, individual .txt caption files are written."
        ),
    )
    group.add_argument(
        "-p", "--prefix", default="",
        help="Prefix all .txt caption files with the specified text. "
             "If the prefix matches a tag, the tag will not be repeated.",
    )
    group.add_argument(
        "-s", "--shuffle", action="store_true",
        help="Shuffle tag strings."
    )
    group.add_argument(
        "-u", "--underscores", action="store_true",
        help="Do not convert underscores to spaces.",
    )
    group.add_argument(
        "-P", "--prompt", action="store_true",
        help="Escape prompt syntax such as parenthesis.",
    )
    group.add_argument(
        "-a", "--alias", action="append", nargs=2, default=[],
        metavar=("OLD", "NEW"),
        help="Change the name of a tag."
    )
    group.add_argument(
        "-A", "--aliases", action="append", default=[],
        metavar="PATH",
        help="Path to tag alias file, with one space-separated alias per line. "
             "May be specified multiple times."
    )
    group.add_argument(
        "-B", "--category-prefix", action="append", nargs=2, default=[],
        metavar=("CATEGORY", "PREFIX"),
        help="Define a prefix appended to all tags with the specified category. "
             "May be specified multiple times."
    )

    group = parser.add_argument_group("classification")
    group.add_argument(
        "-m", "--metric", default="f1.0@0.1",
        metavar="METRIC",
        help="Calibration metric. (Default: f1.0@0.1)",
    )
    group.add_argument(
        "-i", "--implications", choices=IMPLICATION_MODES, default="inherit",
        metavar="MODE",
        help="Automatically apply implications. (Default: inherit)",
    )
    group.add_argument(
        "-x", "--exclude-tag", action="append", default=[],
        metavar="TAG",
        help="Exclude the specified tag. May be specified multiple times.",
    )
    group.add_argument(
        "-X", "--exclude-tags", action="append", default=[],
        metavar="PATH",
        help="Load a list of tags to exclude from the specified file. "
             "May be specified multiple times.",
    )
    group.add_argument(
        "-C", "--exclude-category", action="append", default=[],
        metavar="CATEGORY",
        help="Exclude the specified category of tags. May be specified multiple times.",
    )
    group.add_argument(
        "-g", "--exclusive-group", action="append", nargs="+", default=[],
        metavar=("[[[!]TAG ...] [!]TAG:] TAG TAG", "TAG"),
        help="Define a group of mutually-exclusive tags with an optional precondition. "
             "May be specified multiple times.",
    )
    group.add_argument(
        "-G", "--exclusive-groups", action="append", default=[],
        metavar="PATH",
        help="Load a list of mutually-exclusive groups from the specified file, one per line. "
             "May be specified multiple times.",
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
        help="Batch size.",
    )
    group.add_argument(
        "-w", "--workers", type=int, default=default_workers,
        metavar="N_WORKERS",
        help=
            "Number of dataloader workers, capped to half the number of images. "
            "0 loads all images serially in-process. "
            "-1 creates one loader per cpu core, up to 16 or one per 4 images. "
            "-2 creates one loader per cpu core. "
            f"(Default: {default_workers})"
    )
    group.add_argument(
        "--no-shm", action="store_true",
        help="Disable shared memory between workers.",
    )
    group.add_argument(
        "-V", "--varlen", action="store_true",
        help="Use optimized varlen attention. (Requires flash attention support.)",
    )
    group.add_argument(
        "-S", "--seqlen", type=int, default=1024,
        help="NaFlex sequence length. (Default: 1024)",
    )
    group.add_argument(
        "-d", "--device", default=default_device,
        metavar="DEVICE",
        help=f"Torch device. (Default: {default_device})",
    )
    group.add_argument(
        "-c", "--compile", action="store_true",
        help="Compile the model for maximum performance.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 64 <= args.seqlen <= 2048:
        parser.error("--seqlen must be between 64 and 2048")

    exclude_tags: set[str] = set(args.exclude_tag)
    for path in args.exclude_tags:
        with open(path, "r", encoding="utf-8") as exclude_file:
            exclude_tags.update(parse_list(exclude_file))

    aliases = dict(args.alias)
    for path in args.aliases:
        with open(path, "r", encoding="utf-8") as aliases_file:
            aliases.update(parse_aliases(aliases_file))

    exclusive_groups: list[ExclusiveGroup] = [
        ExclusiveGroup.parse(group)
        for group in args.exclusive_group
    ]

    for path in args.exclusive_groups:
        with open(path, "r", encoding="utf-8") as groups_file:
            exclusive_groups.extend(ExclusiveGroup.parse_file(groups_file))

    paths = list(paths_iter(args.paths, args.recursive))
    if not paths:
        parser.error("no classifiable files found")

    print(f"Loading {repr(args.model)} ...", end="", file=sys.stderr)
    model = load_model(args.model, legacy_metadata_dir=args.metadata)
    model.image_config.max_seqlen = args.seqlen
    print(f" {len(model.labels)} tags.", file=sys.stderr)

    if (not args.extension and not args.no_default_extensions):
        default_extensions = "extensions/" + os.path.splitext(os.path.basename(args.model))[0]
        if os.path.isdir(default_extensions):
            args.extension.append(default_extensions)

    if args.extension:
        print(f"Loading extensions ...", file=sys.stderr)
        for ext in model.load_extensions(Extension.discover(args.extension)):
            print(f"  {repr(ext.path)}: {repr(ext.label.label)} ({ext.label.category})")

    if args.device != "cpu":
        print(f"Transferring to device {repr(args.device)} ...", end="", file=sys.stderr)
        model = model.to(device=args.device)
        print(f" done.", file=sys.stderr)

    if args.compile:
        model.compile(mode="max-autotune-no-cudagraphs")

    rewrite: dict[str, Any] = {
        "aliases": aliases,
        "spaces": not args.underscores,
        "escape": args.prompt,
        "prefix": args.prefix,
        "prefixes": dict(args.category_prefix),
        "shuffle": args.shuffle,
    }

    match args.output:
        case None:
            csv_output: TextIO | None = None
        case "-":
            csv_output = sys.stdout
        case _:
            csv_output = open(
                args.output, "w",
                buffering=1024 * 1024,
                encoding="utf-8",
                newline="",
            )

    calibration: Calibration | None = None
    if csv_output is None:
        print(f"Calibrating ...", end="", file=sys.stderr)
        calibration = model.calibrate(args.metric)
        print(f" done.", file=sys.stderr)

    print(f"Launching dataloader ...", end="", file=sys.stderr)
    loader = Loader(
        args.batch_size, model.image_config, args.workers,
        max_workers=Loader.heuristic_max_workers(args.workers, len(paths), args.batch_size),
        share_memory=not args.no_shm
    )
    print(f" {loader.n_workers} worker{'' if loader.n_workers == 1 else 's'}.", file=sys.stderr)

    try:
        process_batched(
            model=model,
            loader=loader,
            paths=paths_iter(args.paths, args.recursive),
            batch_size=args.batch_size,
            calibration=calibration,
            csv_output=csv_output,
            implications=args.implications,
            exclude_categories=set(args.exclude_category),
            exclude_tags=exclude_tags,
            exclusive_groups=exclusive_groups,
            rewrite=rewrite,
            varlen_attn=args.varlen,
            on_error=lambda path, ex: print(f"{repr(path)}: {ex}", file=sys.stderr),
        )
    finally:
        if csv_output is not None and csv_output is not sys.stdout:
            csv_output.close()

        loader.shutdown()

if __name__ == "__main__":
    main()
