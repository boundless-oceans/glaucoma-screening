"""Standard-library guards for the official-only R16 training boundary."""
from pathlib import PurePosixPath


def validate_roles(train, valid, expected=(51396, 1657, 5849, 191)):
    identities=[]
    for rows,split,role,folder in [(train,'train','r16_official_train','train'),
                                  (valid,'val','r16_official_valid','valid')]:
        ids=[];paths=[]
        for row in rows:
            if row.get('official_split')!=split or row.get('role')!=role:
                raise ValueError('Official split/role violation')
            name=str(row.get('image_id',''))
            relative=str(row.get('relative_path',''));p=PurePosixPath(relative)
            if not name or name in ('None','nan') or not p.parts or p.parts[0]!=folder or p.is_absolute() or '..' in p.parts or '\\' in relative:
                raise ValueError('Invalid image ID/path')
            if str(row.get('label')) not in ('0','1','0.0','1.0'):
                raise ValueError('Binary labels required')
            ids.append(name);paths.append(relative)
        if len(set(ids))!=len(ids) or len(set(paths))!=len(paths):
            raise ValueError('Duplicate image ID/path')
        identities.append(set(ids))
    if identities[0]&identities[1]:raise ValueError('Official train/valid ID overlap')
    counts=(len(train),sum(int(float(r['label'])) for r in train),
            len(valid),sum(int(float(r['label'])) for r in valid))
    if counts!=tuple(expected):raise ValueError('Official counts/positives changed: '+str(counts))
    return dict(zip(('train_n','train_positive','valid_n','valid_positive'),counts))
