from churn.data import load_raw
from churn.eda import summarize


def main() -> None:
    print(summarize(load_raw()))


if __name__ == "__main__":
    main()
