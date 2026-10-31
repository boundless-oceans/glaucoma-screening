"""Bundle CLI; --help is stdlib-only. GPU inference runs on Linux server only."""
from pathlib import Path
import sys

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str((here if (here/'src').is_dir() else here.parent)/'src'))
    from r16_deploy.runtime import main
    raise SystemExit(main())
