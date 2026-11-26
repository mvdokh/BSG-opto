"""Unified command-line interface for running any bsg_opto experiment."""
import argparse
import sys
from .experiments import EXPERIMENTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python run.py',
        description='Optogenetic stimulation experiments on the Lindén et al. (2022) balanced spinal network.',
    )
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List all available experiments and their descriptions')

    for name, mod in EXPERIMENTS.items():
        p = sub.add_parser(name, help=mod.DESCRIPTION, description=mod.DESCRIPTION)
        mod.add_args(p)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'list':
        print('Available experiments:\n')
        for name, mod in EXPERIMENTS.items():
            print(f'  {name:<22s} {mod.DESCRIPTION}')
        print('\nRun `python run.py <experiment> --help` for options.')
        return 0
    mod = EXPERIMENTS[args.command]
    kwargs = {k: v for k, v in vars(args).items() if k != 'command'}
    mod.run(**kwargs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
