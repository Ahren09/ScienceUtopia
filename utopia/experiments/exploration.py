"""Portable exploration experiments."""


def main(argv=None):
    from utopia.experiments.common import main_for
    return main_for("exploration", argv)


if __name__ == "__main__":
    main()
