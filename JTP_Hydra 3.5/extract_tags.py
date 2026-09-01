if __name__ == "__main__":
    import argparse
    import csv
    from sys import stdout

    from hydra.label import Label
    from hydra.model import load_model

    default_model = "models/JTP-Hydra-3.5.safetensors"

    parser = argparse.ArgumentParser(
        description="Hydra Tag Extractor",
        allow_abbrev=False,
    )

    parser.add_argument(
        "model", default=default_model,
        metavar="PATH",
        help=f"Path to model file. (Default: {default_model})",
    )

    parser.add_argument(
        "-o", "--output", default="-",
        help=f"Path for CSV output, or '-' for standard output. (Default: standard output)"
    )

    args = parser.parse_args()

    if args.output == "-":
        output = stdout
    else:
        output = open(args.output, "w", newline="", encoding="utf-8")

    model = load_model(args.model, metadata_only=True)

    writer = csv.writer(output)
    writer.writerow(("tag", "category", "implications"))

    for label in model.labels:
        writer.writerow((label.label, label.category, " ".join(label.implies)))
