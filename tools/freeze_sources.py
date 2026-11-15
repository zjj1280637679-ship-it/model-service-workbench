"""Record a declared source set without claiming validation or live execution."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('runtime/minimax_h3', 'tools', 'tests', 'docs', 'knowledge', 'notebooks', 'research', 'evidence/research-2026-11-15', 'evidence/conditions-2026-11-15', 'evidence/runtime-conditions-2026-11-15', '.github')
ROOT_FILES = ('README.md', 'AGENTS.md', 'NOTICE.md', 'VERSION', 'requirements-test.txt', 'pytest.ini', '.gitignore', '.gitattributes',
              'evidence/acceptance-2026-11-15.json', 'evidence/import-provenance.json')

def main():
    paths = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for directory in DIRECTORIES:
        paths.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
                     and p.suffix not in {'.pyc', '.pyo'} and 'runtime' not in p.relative_to(ROOT / directory).parts)
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    record = {'schema_version': 1, 'files': files,
              'scope': 'Declared source, knowledge and historical evidence inputs; excludes repository-validation output and this manifest itself. Hash match is not proof of semantic validity.',
              'normalization': 'Repository text files are LF; hashes cover exact current bytes.'}
    target = ROOT / 'evidence/source-manifest.json'
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps({'recorded_files':len(files),'manifest':target.relative_to(ROOT).as_posix()}))

if __name__ == '__main__':
    main()
