"""Run ERM baseline on MetaShift top-50 filtered dataset."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SPUME-master"))
sys.path.insert(0, str(ROOT / "scripts" / "metashift"))

from train_erm_baseline import train, ERMResNet, build_loaders, evaluate, AverageMeter
from train_erm_baseline import ROOT as _orig_root

# Override paths
import train_erm_baseline as teb
teb.METADATA_CSV = ROOT / "datasets/metashift/metadata_metashift_catdog_top50.csv"
teb.OUT_DIR = ROOT / "analysis/metashift/erm_top50"

class Args:
    lr = 1e-3
    batch_size = 64
    num_epochs = 50
    num_workers = 4
    seed = 100
    mixup_alpha = 0.0

args = Args()
# Make sure OUT_DIR is set globally in the module
teb.train(args)
