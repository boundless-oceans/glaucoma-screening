"""Bind a completed same-runtime audit as a distinct paired experiment control."""
import json
from pathlib import Path,PurePosixPath

from .contracts import read_csv,sha,probability


def load_reference(folder,config,runtime,rows):
    folder=Path(folder).resolve()
    complete=json.loads((folder/'complete.json').read_bytes())
    for name,digest in complete['files'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:raise ValueError('Unsafe audit path')
        if sha(folder/name)!=digest:raise ValueError('Audit artifact changed: '+name)
    if not {'registration.json','predictions.csv','results.json','status.json'}<=set(complete['files']):
        raise ValueError('Incomplete audit files')
    registration=json.loads((folder/'registration.json').read_bytes())
    result=json.loads((folder/'results.json').read_bytes())
    status=json.loads((folder/'status.json').read_bytes())
    if (status['status']!='completed' or result['status']!='completed' or result['mode']!='baseline-audit'
            or result['n']!=5849 or complete['fingerprint']!=result['fingerprint']
            or registration['fingerprint']!=result['fingerprint']):raise ValueError('Incomplete/inconsistent audit')
    if registration['runtime']!=runtime:raise ValueError('Audit runtime differs')
    old=registration['config']
    for key in ('policy','official_valid_sha256','reference_sha256','baseline_guard'):
        if old[key]!=config[key]:raise ValueError('Audit data/policy differs: '+key)
    for name,item in config['weights'].items():
        if old['weights'][name]['sha256']!=item['sha256']:raise ValueError('Audit weights differ')
    for name,digest in config['source_sha256'].items():
        if name.startswith('src/glaucoma/') or name in ('src/r16_tta/imaging.py','src/r16_tta/pipeline.py','valid_manifest.csv'):
            if old['source_sha256'].get(name)!=digest:raise ValueError('Audit inference source differs: '+name)
    for metric,limit in (('auc_probability','max_abs_auc_delta'),('ap_probability','max_abs_ap_delta')):
        if abs(result['original_metrics'][metric]-result['reference_metrics'][metric])>config['baseline_guard'][limit]:
            raise ValueError('Audited aggregate metrics differ too much from archive')
    # All archived-probability outliers must have explicit repeat/helper diagnostics.
    for item in result['outliers']:
        if item['report'] not in complete['files']:raise ValueError('Unbound outlier diagnostic')
        diagnostic=json.loads((folder/item['report']).read_bytes())
        if not diagnostic['repeat_equal'] or not diagnostic['helper_equal']:
            raise ValueError('Unresolved audit inference inconsistency')
    predictions=read_csv(folder/'predictions.csv');by_id={r['image_id']:r for r in predictions}
    if len(predictions)!=5849 or len(by_id)!=5849 or set(by_id)!={r['image_id'] for r in rows}:
        raise ValueError('Audit prediction IDs differ')
    controls={}
    for row in rows:
        record=by_id[row['image_id']]
        if int(record['label'])!=row['label'] or probability(record['reference_probability'])!=row['reference_probability']:
            raise ValueError('Audit labels/archive reference differ')
        controls[row['image_id']]=probability(record['probability'])
    actual_outliers={r['image_id'] for r in rows if abs(controls[r['image_id']]-r['reference_probability'])>config['baseline_guard']['max_abs_probability_delta']}
    if actual_outliers!={r['image_id'] for r in result['outliers']}:raise ValueError('Audit outlier coverage differs')
    provenance=dict(kind='same_runtime_audited_original',audit_directory=str(folder),
                    complete_sha256=sha(folder/'complete.json'),predictions_sha256=sha(folder/'predictions.csv'),
                    archival_strict_reproduction=result['reproduction']['baseline_reproduced'],
                    archive_is_context_not_paired_control=True)
    return controls,provenance
