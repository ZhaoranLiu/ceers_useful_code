import argparse
from .core import process_rate_files, generate_sky_background, run_bkg_subtraction, full_pipeline, run_cont_subtraction, combined_reduce_and_generate_sw, perform_astrometric_calculations, plot_sources_on_footprint, spectra_extration, plot_spectra


def wcs_flat_fielding():
    process_rate_files()

def sky_background(data_type):
    generate_sky_background(data_type)

def bkg_subtraction():
    run_bkg_subtraction()

def super_background(data_type):
    full_pipeline(data_type)

def cont_subtraction(L_box, L_mask):
    run_cont_subtraction(L_box, L_mask)

def reduce_sw():
    combined_reduce_and_generate_sw()
    
def footprint_check():
    plot_sources_on_footprint()
    
def astrometric_calculations():
    perform_astrometric_calculations()

def extration():
    spectra_extration()
    
def plot():
    plot_spectra()
    
def main():
    parser = argparse.ArgumentParser(description='Process grism data.')
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('wcs_flat_fielding', help='WCS corrections + Flat Fielding subtraction')
    parser_sky_bg = subparsers.add_parser('sky_background', help='Generate Sky Background')
    parser_sky_bg.add_argument('--data_type', choices=['GRISMC', 'GRISMR', 'BOTH'], required=True, help='Type of data (GRISMC, GRISMR, BOTH)')

    subparsers.add_parser('bkg_subtraction', help='Background subtraction')

    parser_super_bg = subparsers.add_parser('super_background', help='Run the full pipeline')
    parser_super_bg.add_argument('--data_type', choices=['GRISMC', 'GRISMR', 'BOTH'], required=True, help='Type of data (GRISMC, GRISMR, BOTH)')

    parser_cont_sub = subparsers.add_parser('cont_subtraction', help='Continuum subtraction and EMLINE image generation')
    parser_cont_sub.add_argument('--L_box', type=int, required=True, help='Size of the median filter box')
    parser_cont_sub.add_argument('--L_mask', type=int, required=True, help='Size of the mask in the median filter')
    
    subparsers.add_parser('reduce_sw', help='Reduce and generate SW files')
    subparsers.add_parser('astrometric', help='Perform Astrometric Calculations')
    subparsers.add_parser('check_footprint', help='Check overlap of sources and observations footprint')
    subparsers.add_parser('spectra_extration', help='Run all steps for astrometric calculations, plotting, and spectra extraction')
    subparsers.add_parser('plot_spectra', help='Plot and save extracted spectra')
    
    
    args = parser.parse_args()

    if args.command == 'wcs_flat_fielding':
        wcs_flat_fielding()
    elif args.command == 'sky_background':
        sky_background(args.data_type)
    elif args.command == 'bkg_subtraction':
        bkg_subtraction()
    elif args.command == 'super_background':
        super_background(args.data_type)
    elif args.command == 'cont_subtraction':
        cont_subtraction(args.L_box, args.L_mask)
    elif args.command == 'reduce_sw':
        reduce_sw()
    elif args.command == 'astrometric':
        astrometric_calculations()
    elif args.command == 'check_footprint':
        footprint_check()
    elif args.command == 'plot_spectra':
        plot_spectra()
    elif args.command == 'spectra_extration':
        extration()
    


if __name__ == '__main__':
    main()

