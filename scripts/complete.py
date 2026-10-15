import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quill.data.agda.reader import parse_data
from quill.nn.inference import Inferer

from agda_source import (
    export_agda_file,
    find_source_holes,
    iter_holes,
    replace_source_holes,
    source_name,
)


def add_body_hole_queries(file_json: dict) -> int:
    added = 0
    for entry in file_json.get('scope-local', []):
        definition = entry.get('definition', {})
        if definition.get('tag') != 'Function':
            continue
        for clause in definition.get('clauses', []):
            body = clause.get('body', {})
            if body.get('tag') != 'UnsolvedMetavariable':
                continue
            entry.setdefault('holes', []).append({
                '_source_hole': True,
                'ctx': {'pretty': '', 'telescope': []},
                'goal': entry['type'],
                'term': {'tag': 'UnsolvedMetavariable'},
                'premises': [],
            })
            added += 1
    return added


def apply_predictions(source: str, file_json: dict, predictions: list[list[str]]) -> tuple[str, int]:
    prediction_index = 0
    replacements = []

    for hole in iter_holes(file_json):
        if prediction_index >= len(predictions):
            raise ValueError('Model returned fewer predictions than the input has holes.')

        suggestions = predictions[prediction_index]
        prediction_index += 1
        term = hole.get('term', {})
        if hole.get('_source_hole'):
            replacements.append(source_name(suggestions[0]) if suggestions else None)

    if prediction_index != len(predictions):
        raise ValueError('Model returned more predictions than the input has holes.')
    return replace_source_holes(source, replacements)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fill explicit holes in an Agda source file with top-ranked premises.'
    )
    parser.add_argument('input_file', type=Path, help='Input Agda source file')
    parser.add_argument('output_file', type=Path, help='Output completed Agda source file')
    args = parser.parse_args()
    if args.output_file.suffix != '.agda':
        parser.error('output_file must have the .agda extension')

    config_path = ROOT / 'data' / 'config.json'
    weights_path = ROOT / 'data' / 'model.pt'
    with config_path.open('r') as f:
        config = json.load(f)
    source = args.input_file.read_text()
    file_json = export_agda_file(args.input_file)
    source_holes = len(find_source_holes(source))
    query_count = add_body_hole_queries(file_json)
    if source_holes != query_count:
        raise ValueError(
            f'Found {source_holes} source hole(s), but could build '
            f'{query_count} definition-body query/queries.'
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inferer = Inferer(config['model_config'], device)
    inferer.load(str(weights_path), strict=True, map_location=device)
    predictions = inferer.select_premises(parse_data(file_json, validate=True))
    completed_source, completed = apply_predictions(source, file_json, predictions)

    args.output_file.write_text(completed_source)
    print(f'Completed {completed} hole(s); wrote {args.output_file}.')


if __name__ == '__main__':
    main()
