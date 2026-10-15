import argparse
from pathlib import Path

from agda_source import export_agda_file, punch_top_level_bodies


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate and prepare an Agda file that already has explicit holes.'
    )
    parser.add_argument('input_file', type=Path, help='Input Agda source file')
    parser.add_argument('output_file', type=Path, help='Output Agda source file')
    args = parser.parse_args()
    if args.output_file.suffix != '.agda':
        parser.error('output_file must have the .agda extension')

    source = args.input_file.read_text()
    export_agda_file(args.input_file)
    punched_source, punched = punch_top_level_bodies(source)
    if not punched:
        raise ValueError('No module-level definition bodies found to punch.')

    args.output_file.write_text(punched_source)
    print(f'Punched {punched} definition body/bodies; wrote {args.output_file}.')


if __name__ == '__main__':
    main()
