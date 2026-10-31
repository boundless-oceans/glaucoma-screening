"""Server-only original-R16 TTA check/evaluation; never trains; test export is explicit."""
from pathlib import Path
import sys

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    source=here if (here/'src').is_dir() else here.parent
    sys.path.insert(0,str(source/'src'))
    from r16_tta.evaluate import main
    raise SystemExit(main())
