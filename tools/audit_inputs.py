#!/usr/bin/env python3
"""Non-mutating inventory of probable external references in staged LAMMPS scripts.
This is a heuristic audit, not a LAMMPS parser: inspect variables and generated paths manually.
"""
import argparse
import csv
import re
from pathlib import Path

COMMANDS = re.compile(r'^\s*(read_data|read_restart|include|pair_coeff|molecule|read_dump|write_data|write_restart|restart)\s+(.+?)\s*$', re.I)
QUOTED_PATHS = re.compile(r'(?<![\w])(?:/[^\s"\']+|\.\.?/[^\s"\']+|[^\s"\']+\.(?:data|lmp|pb|in|restart|rst))')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repository', nargs='?', default='.')
    ap.add_argument('--output', default='INPUT_DEPENDENCY_AUDIT.tsv')
    args = ap.parse_args()
    root = Path(args.repository).resolve()
    rows = []
    candidates = [p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts
                  and (p.suffix.lower() in ('.in', '.lmp') or p.name in ('melt.in', 'in.lammps', 'in.lmp'))]
    for p in candidates:
        try:
            lines = p.read_text(errors='replace').splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            part = line.split('#', 1)[0].strip()
            m = COMMANDS.match(part)
            if not m:
                continue
            command, args_text = m.groups()
            refs = QUOTED_PATHS.findall(args_text)
            if command in ('read_data', 'read_restart', 'include', 'molecule', 'read_dump') and not refs:
                refs = [args_text.split()[0]] if args_text.split() else []
            if not refs:
                rows.append([str(p.relative_to(root)), i, command, args_text, '', 'review manually'])
            for ref in refs:
                if '$' in ref or '*' in ref or '${' in ref:
                    status = 'variable/wildcard: review'
                elif ref.startswith('/'):
                    status = 'absolute HPC path: review (exists on this host)' if Path(ref).exists() else 'absolute HPC path: missing here'
                else:
                    status = 'relative path exists' if (p.parent / ref).exists() else 'relative path missing or dynamically generated'
                rows.append([str(p.relative_to(root)), i, command, args_text, ref, status])
    out = root / args.output
    with out.open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['input_file', 'line', 'command', 'arguments', 'possible_reference', 'status'])
        w.writerows(rows)
    print(f'Inspected {len(candidates)} candidate input files; wrote {len(rows)} reference rows to {out}')


if __name__ == '__main__':
    main()
