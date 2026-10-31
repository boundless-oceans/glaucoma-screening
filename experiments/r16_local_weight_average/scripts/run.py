"""Linux server stages. --help imports no framework; no training/test export."""
from pathlib import Path
import sys

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str((here if (here/'src').is_dir() else here.parent)/'src'))
    from r16_average.runtime import main
    raise SystemExit(main())
