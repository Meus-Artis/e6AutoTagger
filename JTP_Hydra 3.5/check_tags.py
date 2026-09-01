if __name__ == "__main__":
    import argparse
    import csv
    import os
    import re
    from sys import stderr, stdout

    from hydra.label import Label, Rewriter, parse_aliases
    from hydra.model import Extension, load_model

    default_model = "models/JTP-Hydra-3.5.safetensors"

    parser = argparse.ArgumentParser(
        description="Hydra Tag Checker",
        allow_abbrev=False,
    )

    parser.add_argument(
        "-M", "--model", default=default_model,
        metavar="PATH",
        help=f"Path to model file. (Default: {default_model})",
    )
    parser.add_argument(
        "-D", "--metadata", default="./data",
        metavar="PATH",
        help="Metadata directory for legacy JTP-3 models. (Default: ./data)",
    )
    parser.add_argument(
        "-e", "--extension", action="append", default=[],
        metavar="PATH",
        help=(
            "Path to extension. May be specified multiple times. "
            "If a directory is specified, all extensions in the specified directory are loaded. "
            "(Default: extensions/<model_name>)"
        ),
    )
    parser.add_argument(
        "-E", "--no-default-extensions", action="store_true",
        help="Do not load extensions by default.",
    )

    parser.add_argument(
        "-u", "--underscores", action="store_true",
        help="Do not convert underscores to spaces.",
    )
    parser.add_argument(
        "-P", "--prompt", action="store_true",
        help="Escape prompt syntax such as parenthesis.",
    )
    parser.add_argument(
        "-a", "--alias", action="append", nargs=2, default=[],
        metavar=("OLD", "NEW"),
        help="Change the name of a tag."
    )
    parser.add_argument(
        "-A", "--aliases", action="append", default=[],
        metavar="PATH",
        help="Path to tag alias file, with one space-separated alias per line. "
             "May be specified multiple times."
    )
    parser.add_argument(
        "-B", "--category-prefix", action="append", nargs=2, default=[],
        metavar=("CATEGORY", "PREFIX"),
        help="Define a prefix appended to all tags with the specified category. "
             "May be specified multiple times."
    )

    parser.add_argument(
        "-x", "--exclude-tag", action="append", default=[],
        metavar="TAG",
        help="Exclude the specified tag. May be specified multiple times.",
    )
    parser.add_argument(
        "-X", "--exclude-tags", action="append", default=[],
        metavar="PATH",
        help="Load a list of tags to exclude from the specified file. "
             "May be specified multiple times.",
    )
    parser.add_argument(
        "-C", "--exclude-category", action="append", default=[],
        metavar="CATEGORY",
        help="Exclude the specified category of tags. May be specified multiple times.",
    )

    parser.add_argument(
        "-c", "--csv", nargs="+",
        metavar=("NAME", "COUNT THRESHOLD"),
        help="Interpret the input as a CSV file and use the specified column for the names of valid tags. "
             "If specified, only consider tags valid if the value in the count column meets the threshold."
    )

    parser.add_argument(
        "path",
        metavar="PATH",
        help=f"Path to text file containing the list of valid tags."
    )

    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as file:
        if args.csv is not None:
            match len(args.csv):
                case 1:
                    threshold = None
                case 3:
                    threshold = int(args.csv[2])
                case _:
                    parser.error("--csv requires either 1 or 3 arguments")

            valid_tags = {
                row[args.csv[0]]
                for row in csv.DictReader(file)
                if threshold is None or int(row[args.csv[1]]) >= threshold
            }
        elif args.underscores:
            valid_tags = set(file.read().split())
        else:
            valid_tags = {
                tag
                for item in re.split(r"[,\t\r\n]+", file.read())
                if (tag := item.strip())
            }

    exclude_categories = set(args.exclude_category)

    exclude_tags: set[str] = set(args.exclude_tag)
    for path in args.exclude_tags:
        with open(path, "r", encoding="utf-8") as exclude_file:
            exclude_tags.update(exclude_file.read().split())

    aliases = dict(args.alias)
    for path in args.aliases:
        with open(path, "r", encoding="utf-8") as aliases_file:
            aliases.update(parse_aliases(aliases_file.read()))

    model = load_model(args.model, legacy_metadata_dir=args.metadata, metadata_only=True)

    if (not args.extension and not args.no_default_extensions):
        default_extensions = "extensions/" + os.path.splitext(os.path.basename(args.model))[0]
        if os.path.isdir(default_extensions):
            args.extension.append(default_extensions)

    if args.extension:
        model.load_extensions(Extension.discover(args.extension), metadata_only=True)

    rewriter = Rewriter.create(
        aliases=aliases,
        spaces=not args.underscores,
        escape=args.prompt,
        prefixes=dict(args.category_prefix)
    )

    mapping: list[tuple[str, str]] = []
    n_auto = 0

    for label in model.labels:
        if (
            label.category in exclude_categories
            or label.label in exclude_tags
        ):
            continue

        rewritten = rewriter.rewrite_fn(label)
        if rewritten in valid_tags:
            mapping.append((label.label, rewritten))
            continue

        if label.category == "general":
            candidate = Label(
                label.label
                    .replace("another", "partner")
                    .replace("elderly", "old")
                    .replace("femboy", "girly")
                    .replace("vulva", "pussy")
                    .replace("vaginal_fluids", "pussy_juice")
                    .replace("vaginal_fluid", "pussy_juice")
                    .replace("vaginal_squirting", "pussy_ejaculation")
                    .replace("vagina", "pussy"),
                "general", [], None
            )

            rewritten = rewriter.rewrite_fn(candidate)
            if rewritten in valid_tags:
                mapping.append((label.label, rewritten))
                n_auto += 1

                print(f"{label.label} {candidate.label}")
                continue

        print(f"{label.label}")

    if n_auto:
        print(f"AUTOMATIC: {n_auto}", file=stderr)

    for keys, value in Rewriter.conflicts(mapping):
        print("CONFLICT: " + " ".join(keys) + f" > {repr(value)}", file=stderr)
