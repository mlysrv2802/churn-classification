import argparse

from churn.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and manage saved churn models.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show every saved version")
    rollback = commands.add_parser("rollback", help="make an existing version the production model")
    rollback.add_argument("version", help="version id from `list`")
    args = parser.parse_args()

    registry = ModelRegistry()
    if args.command == "list":
        versions = registry.list_versions()
        if not versions:
            print("No saved models yet. Run `python scripts\\train.py`.")
        print(f"{'':2}{'version':22}{'trained_at':27}{'roc_auc':>8}{'cutoff':>8}")
        for v in versions:
            marker = "*" if v["is_production"] else ""
            print(
                f"{marker:2}{v['version']:22}{v['trained_at']:27}"
                f"{v['test_metrics']['roc_auc']:8.4f}{v['threshold']:8.2f}"
            )
        print("\n* = production")
    else:
        try:
            registry.rollback(args.version)
        except ValueError as error:
            raise SystemExit(f"Rollback failed: {error}. Run `list` to see valid versions.")
        print(f"Production is now {args.version}. Restart the API to load it.")


if __name__ == "__main__":
    main()
