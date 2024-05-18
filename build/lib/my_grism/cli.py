# my_grism/cli.pys

import argparse
from .core import process_rate_files

def wcs_flat_fielding():
    all_rate_list = process_rate_files()
    print(all_rate_list)

def main():
    parser = argparse.ArgumentParser(description='Process grism data.')
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('wcs_flat_fielding', help='WCS corrections + Flat Fielding subtraction')

    args = parser.parse_args()

    if args.command == 'wcs_flat_fielding':
        wcs_flat_fielding()

if __name__ == '__main__':
    main()



