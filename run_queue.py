# run_queue.py
# Runs training jobs one at a time (the 4 GB GPU cannot hold several). Safe to rerun:
#   - jobs with a DONE marker are skipped
#   - jobs with saved epochs are continued with --resume
#   - a STOP file in runs/ stops the current job after its epoch is saved, and the queue
#     does not start the next job. Delete STOP and rerun this script to continue.
#   - a PAUSE file in runs/ idles the current job until the file is removed.
#
# Usage:  python run_queue.py ce          (encoder ablation, CE stage)
import glob
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE_DIR, 'runs')

def ce_job(enc):
    return f'ce_{enc}_s0', ['--feat_dir', f'data/coco/timm_features/{enc}', '--loss', 'ce', '--seed', '0']


def scst_job(name, *extra):
    # SCST stage on SE-ResNet-50 (decided before its CE results existed; see DECISIONS.md).
    # Every SCST variant trains the full 10 epochs (--patience 10 disables early stopping),
    # matching the paper; see DECISIONS.md for why the no-normalization run needed this.
    return name, ['--feat_dir', 'data/coco/timm_features/seresnet50', '--init', 'runs/ce_seresnet50_s0/best.pt',
                  '--seed', '0', '--patience', '10', *extra]


QUEUES = {
    'ce': [ce_job(enc) for enc in ['resnet50', 'seresnet50', 'resnet50d']],
    # Everything, in priority order: the loss-ablation rows the paper cannot do without come
    # first, the two stabilizer ablations last (first to cut if time runs out).
    'main': [
        ce_job('resnet50'),
        ce_job('seresnet50'),
        scst_job('scst_mixed_se_s0', '--loss', 'mixed'),
        scst_job('scst_pure_se_s0', '--loss', 'scst'),
        ce_job('resnet50d'),
        scst_job('scst_mixed_nonorm_se_s0', '--loss', 'mixed', '--no_norm_adv'),
        scst_job('scst_mixed_noreset_se_s0', '--loss', 'mixed', '--keep_optimizer'),
    ],
}


def main():
    queue = QUEUES[sys.argv[1]]
    for name, job_args in queue:
        out = os.path.join(RUNS, name)
        if os.path.exists(os.path.join(out, 'DONE')):
            print(f'[queue] {name}: done, skipping', flush=True)
            continue
        if os.path.exists(os.path.join(RUNS, 'STOP')):
            print('[queue] runs/STOP present; not starting further jobs', flush=True)
            return
        resume = ['--resume'] if glob.glob(os.path.join(out, 'epoch_*.pt')) else []
        print(f'[queue] {name}: {"resuming" if resume else "starting"}', flush=True)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(RUNS, f'{name}.out'), 'a') as log:
            rc = subprocess.call([sys.executable, 'train_decoder.py', *job_args, '--out', out, *resume],
                                 cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f'[queue] {name}: exited with code {rc}; stopping queue (see runs/{name}.out)', flush=True)
            return
        if not os.path.exists(os.path.join(out, 'DONE')):
            print(f'[queue] {name}: stopped before finishing; rerun the queue to continue', flush=True)
            return
    print('[queue] all jobs done', flush=True)


if __name__ == '__main__':
    main()
