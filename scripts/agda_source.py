import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def export_agda_file(source_file: Path) -> dict:
    source_file = source_file.resolve()
    source = source_file.read_text()
    module_match = re.search(r'(?m)^\s*module\s+([^\s({]+)', source)
    if module_match is None:
        raise ValueError(f'Could not find a module declaration in {source_file}.')
    module_name = module_match.group(1)
    module_path = Path(*module_name.split('.')).with_suffix('.agda')
    conversion_source = source
    has_holes = bool(find_source_holes(source))
    if has_holes:
        conversion_source = re.sub(
            r'(?m)(\{-#\s*OPTIONS\b[^#]*?)\s+--safe\b',
            r'\1',
            source,
        )
        conversion_source = '{-# OPTIONS --allow-unsolved-metas #-}\n' + conversion_source

    with tempfile.TemporaryDirectory(prefix='quill-agda2train-') as temp_dir:
        temp_root = Path(temp_dir)
        output_dir = temp_root / 'json'
        output_dir.mkdir()
        source_root = _find_module_root(source_file, module_path)
        if (
            source_root is not None
            and source_root.joinpath(module_path).resolve() == source_file
            and not has_holes
        ):
            source_to_process = source_file
            include_root = source_root
        elif source_root is not None:
            include_root = temp_root / 'source'
            source_to_process = _create_module_overlay(
                source_root,
                include_root,
                module_path,
                conversion_source,
            )
        else:
            source_to_process = temp_root / module_path
            source_to_process.parent.mkdir(parents=True, exist_ok=True)
            source_to_process.write_text(conversion_source)
            include_root = temp_root

        command = [
            str(Path.home() / '.cabal' / 'bin' / 'agda2train'),
            '--ignore-existing-json',
            '--out-dir',
            str(output_dir),
            '-i',
            str(include_root),
        ]
        command.append(str(source_to_process))
        subprocess.run(
            command,
            check=True,
        )

        expected_file = Path(output_dir) / f'{module_name}.json'
        if expected_file.is_file():
            return json.loads(expected_file.read_text())

        matching_files = []
        for json_file in Path(output_dir).glob('*.json'):
            file_json = json.loads(json_file.read_text())
            if file_json.get('name') == module_name:
                matching_files.append(file_json)
        if len(matching_files) != 1:
            raise RuntimeError(
                f'agda2train did not produce one JSON export for {module_name}.'
            )
        return matching_files[0]


def _find_module_root(source_file: Path, module_path: Path) -> Path | None:
    direct_root = source_file
    for _ in module_path.parts:
        direct_root = direct_root.parent
    if direct_root.joinpath(module_path).resolve() == source_file:
        return direct_root

    ignored_directories = {'.git', '.hg', '.svn', '__pycache__', 'build', 'dist', 'node_modules'}
    search_root = Path.cwd().resolve()
    for directory, child_directories, _ in os.walk(search_root):
        child_directories[:] = [
            child for child in child_directories
            if child not in ignored_directories and not child.startswith('.')
        ]
        candidate_root = Path(directory)
        candidate = candidate_root / module_path
        if candidate.is_file() and candidate.resolve() != source_file:
            return candidate_root
    return None


def _create_module_overlay(
        source_root: Path,
        overlay_root: Path,
        module_path: Path,
        source: str) -> Path:
    source_directory = source_root
    overlay_directory = overlay_root
    parts = module_path.parts

    for part in parts[:-1]:
        overlay_directory.mkdir(parents=True, exist_ok=True)
        for child in source_directory.iterdir():
            if child.name != part:
                (overlay_directory / child.name).symlink_to(
                    child,
                    target_is_directory=child.is_dir(),
                )
        source_directory = source_directory / part
        overlay_directory = overlay_directory / part

    overlay_directory.mkdir(parents=True, exist_ok=True)
    for child in source_directory.iterdir():
        if child.name != parts[-1]:
            (overlay_directory / child.name).symlink_to(
                child,
                target_is_directory=child.is_dir(),
            )

    source_to_process = overlay_directory / parts[-1]
    source_to_process.write_text(source)
    return source_to_process


def iter_holes(file_json: dict):
    for entry in file_json.get('scope-local', []):
        yield from entry.get('holes', [])


def count_unsolved_holes(file_json: dict) -> int:
    return sum(
        hole.get('term', {}).get('tag') == 'UnsolvedMetavariable'
        for hole in iter_holes(file_json)
    )


def punch_top_level_bodies(source: str) -> tuple[str, int]:
    masked = _mask_non_code(source)
    lines = source.splitlines(keepends=True)
    masked_lines = masked.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    edits = []
    punched_lines = []
    for line_index, (line, masked_line, start) in enumerate(zip(lines, masked_lines, offsets)):
        content = masked_line.rstrip('\r\n')
        if not content or content[0].isspace():
            continue
        assignment = _top_level_equals(content)
        if assignment is None or not content[:assignment].strip():
            continue
        body_end = len(content.rstrip())
        if body_end <= assignment + 1:
            continue

        edits.append((start + assignment + 1, start + body_end, ' ?'))
        punched_lines.append(line_index)

    for line_index in punched_lines:
        next_line = line_index + 1
        while next_line < len(lines):
            content = masked_lines[next_line].rstrip('\r\n')
            if content.strip() and not content[0].isspace():
                break
            if content.strip():
                edits.append((offsets[next_line], offsets[next_line] + len(lines[next_line]), ''))
            next_line += 1

    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source, len(punched_lines)


def find_source_holes(source: str) -> list[tuple[int, int]]:
    spans = []
    index = 0
    while index < len(source):
        if source.startswith('--', index):
            newline = source.find('\n', index)
            index = len(source) if newline == -1 else newline + 1
            continue

        if source.startswith('{-', index):
            depth = 1
            index += 2
            while index < len(source) and depth:
                if source.startswith('{-', index):
                    depth += 1
                    index += 2
                elif source.startswith('-}', index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                raise ValueError('Unterminated Agda block comment.')
            continue

        if source[index] == '"':
            index += 1
            while index < len(source):
                if source[index] == '\\':
                    index += 2
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            continue

        if source.startswith('{!', index):
            end = source.find('!}', index + 2)
            if end == -1:
                raise ValueError('Unterminated Agda hole.')
            spans.append((index, end + 2))
            index = end + 2
            continue

        if source[index] == '?':
            before = source[index - 1] if index else ''
            after = source[index + 1] if index + 1 < len(source) else ''
            if not _identifier_character(before) and not _identifier_character(after):
                spans.append((index, index + 1))
        index += 1
    return spans


def replace_source_holes(source: str, replacements: list[str | None]) -> tuple[str, int]:
    spans = find_source_holes(source)
    if len(spans) != len(replacements):
        raise ValueError(
            f'Found {len(spans)} source hole(s), but agda2train reported '
            f'{len(replacements)} unsolved hole(s).'
        )

    completed = 0
    for (start, end), replacement in reversed(list(zip(spans, replacements))):
        if replacement is not None:
            source = source[:start] + replacement + source[end:]
            completed += 1
    return source, completed


def source_name(premise: str) -> str:
    name = re.sub(r'<\d+>$', '', premise)
    if '_' in name.rsplit('.', 1)[-1]:
        return f'({name})'
    return name


def _identifier_character(character: str) -> bool:
    return bool(character) and (character.isalnum() or character in "_'")


def _top_level_equals(line: str) -> int | None:
    depth = 0
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for index, character in enumerate(line):
        if character in '([{':
            stack.append(character)
        elif character in pairs:
            if stack and stack[-1] == pairs[character]:
                stack.pop()
        elif character == '=' and not stack:
            previous = line[index - 1] if index else ''
            following = line[index + 1] if index + 1 < len(line) else ''
            if previous not in '<>!=' and following not in '=>':
                return index
    return None


def _mask_non_code(source: str) -> str:
    masked = list(source)
    index = 0
    comment_depth = 0
    in_string = False
    line_comment = False

    while index < len(source):
        if line_comment:
            if source[index] == '\n':
                line_comment = False
            else:
                masked[index] = ' '
            index += 1
            continue

        if comment_depth:
            if source.startswith('{-', index):
                masked[index:index + 2] = '  '
                comment_depth += 1
                index += 2
            elif source.startswith('-}', index):
                masked[index:index + 2] = '  '
                comment_depth -= 1
                index += 2
            else:
                if source[index] not in '\r\n':
                    masked[index] = ' '
                index += 1
            continue

        if in_string:
            if source[index] == '\\':
                if index + 1 < len(source):
                    masked[index:index + 2] = '  '
                index += 2
            elif source[index] == '"':
                masked[index] = ' '
                in_string = False
                index += 1
            else:
                if source[index] not in '\r\n':
                    masked[index] = ' '
                index += 1
            continue

        if source.startswith('--', index):
            masked[index:index + 2] = '  '
            line_comment = True
            index += 2
        elif source.startswith('{-', index):
            masked[index:index + 2] = '  '
            comment_depth = 1
            index += 2
        elif source[index] == '"':
            masked[index] = ' '
            in_string = True
            index += 1
        else:
            index += 1

    return ''.join(masked)