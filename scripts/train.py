import argparse
import json

from churn.train import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the final churn model.")
    parser.add_argument("--by", choices=["f1", "cost"], default="f1",
                        help="how to choose the cutoff (default: f1)")
    args = parser.parse_args()

    metadata = train_and_save(by=args.by)
    status = "PROMOTED to production" if metadata["promoted"] else "NOT promoted"
    print(f"Saved version {metadata['version']}: {status} ({metadata['promotion_reason']})\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
