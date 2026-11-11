"""Run one authorized experiment phase with log, timeout, and exit receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oem', 'loveda'])
    parser.add_argument('architecture', choices=['resnet18_unet', 'segformer_b0'])
    parser.add_argument('phase', choices=['source', 'adapt', 'evaluate'])
    parser.add_argument('--timeout', type=int, default=21600)
    args = parser.parse_args()
    prepared = ROOT / ('prepared' if args.dataset == 'oem' else 'prepared_loveda')
    manifest = prepared / ('source_manifest.json' if args.phase == 'source' else 'manifest.json')
    config = ROOT / 'configs' / (args.dataset + '.json')
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    output = ROOT / 'runs' / args.dataset / args.architecture
    output.mkdir(parents=True, exist_ok=True)
    stem = f'{args.dataset}_{args.architecture}_{args.phase}'
    receipt = ROOT / 'reports' / f'{stem}_job.json'
    log = ROOT / 'reports' / f'{stem}.log'
    if receipt.exists() or log.exists():
        raise FileExistsError('Preserve existing job; resume requires a versioned launcher/log amendment')
    command = [sys.executable, '-u', str(ROOT / 'runner.py'), '--config', str(config),
               '--manifest', str(manifest), '--output', str(output), '--arch', args.architecture,
               '--phase', args.phase]
    record = {'command': command, 'started_at': time.time(), 'timeout_seconds': args.timeout,
              'status': 'starting', 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'log': str(log)}
    write(receipt, record)
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=ROOT.parent.parent, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record.update(pid=process.pid, status='running')
        write(receipt, record)
        print(json.dumps(record), flush=True)
        try:
            code = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 130
    record.update(exit_code=code, finished_at=time.time(), status='complete' if code == 0 else 'failed')
    write(receipt, record)
    print(json.dumps(record), flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
