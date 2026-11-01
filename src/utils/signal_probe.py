"""Native CUDA warning/checkpoint/restart equivalence through the sealed wrapper."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def run(folder, arm):
    import torch
    from src.train.recovery import configure_execution, stop_requested, save_recovery, load_recovery
    if not torch.cuda.is_available():
        raise RuntimeError('Signal acceptance requires native CUDA')
    configure_execution(True)
    torch.manual_seed(731)
    model = torch.nn.Linear(4, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    identity = dict(fixture='native-signal-v1', updates=12)
    recovery = folder / (arm + '.resume.pt')
    step = 0
    if recovery.exists():
        step = load_recovery(recovery, model=model, optimizer=optimizer,
                             scheduler=scheduler, scaler=scaler, identity=identity)['step']
        if step <= 0:
            raise AssertionError('No completed optimizer update was recovered')
    x = torch.arange(32, device='cuda', dtype=torch.float32).reshape(8, 4) / 32
    while step < 12:
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        step += 1
        if arm != 'baseline' and not recovery.exists():
            (folder / 'ready').touch()
            # Allow the batch shell to deliver its actual warning after readiness.
            for _ in range(400):
                if stop_requested():
                    save_recovery(recovery, model=model, optimizer=optimizer,
                                  scheduler=scheduler, scaler=scaler, identity=identity,
                                  progress=dict(step=step))
                    print(f'Native warning saved update {step}', flush=True)
                    return 75
                time.sleep(.025)
            raise RuntimeError('Batch warning did not reach the native container')
    torch.save(model.state_dict(), folder / (arm + '.pt'))
    return 0


def inspect(folder, destination):
    import torch
    from src.utils.atomic import write_json
    from src.utils.experiment import code_identity
    expected = torch.load(folder / 'baseline.pt', weights_only=True, map_location='cpu')
    for arm in ('USR1', 'TERM'):
        actual = torch.load(folder / (arm + '.pt'), weights_only=True, map_location='cpu')
        if expected.keys() != actual.keys() or any(not torch.equal(expected[k], actual[k]) for k in expected):
            raise RuntimeError('Warning/restart changed final CUDA weights')
    write_json(destination, dict(passed=True, code=code_identity(stage='eval'),
        runtime=os.environ['CREDITPFN_RUNTIME_VERSION'], warnings=['USR1', 'TERM'], exact_resume=True), exclusive=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'inspect']); p.add_argument('folder', type=Path)
    p.add_argument('--arm'); p.add_argument('--receipt', type=Path)
    a = p.parse_args(); a.folder.mkdir(parents=True, exist_ok=True)
    if a.action == 'run':
        raise SystemExit(run(a.folder, a.arm))
    inspect(a.folder, a.receipt)


if __name__ == '__main__':
    main()
