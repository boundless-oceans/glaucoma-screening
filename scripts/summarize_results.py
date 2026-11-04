"""Offline curated experiment ledger. Stdlib only; no models, weights or server."""
import argparse,hashlib,json,math
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def get(obj,key):
    for part in key.split('.'):obj=obj[part]
    return obj


def under(root,name):
    path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):raise ValueError('Artifact path escaped root')
    return path


def number(value):
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid probability metric')
    return value


def collect(catalog,root):
    if catalog['schema']!='experiment-catalog-v1':raise ValueError('Unknown catalog schema')
    result=[];seen=set()
    for item in catalog['entries']:
        if item['id'] in seen:raise ValueError('Duplicate experiment ID')
        seen.add(item['id']);r={k:item[k] for k in ('id','title','decision','reason','platform')}
        r.update(evidence='platform_report_only',metrics=None,report_sha256=None,submission=None,t4_accepted=False)
        if item['report']:
            path=under(root,item['report'])
            if not path.is_file():
                r['evidence']='missing_report';result.append(r);continue
            data=json.loads(path.read_text(encoding='utf-8'))
            for key,expected in item['guards']:
                if get(data,key)!=expected:raise ValueError('Incomplete/wrong report: '+item['id']+'/'+key)
            # If stage hashes exist, validate the report against its completion marker.
            complete=path.parent/'complete.json'
            if complete.is_file():
                marker=json.loads(complete.read_text(encoding='utf-8'));files=marker.get('files',marker)
                if path.name not in files:raise ValueError('Report absent from completion marker')
                expected=files[path.name];expected=expected['sha256'] if isinstance(expected,dict) else expected
                if sha(path)!=expected:raise ValueError('Completion report hash mismatch')
            metric=get(data,item['metrics'])
            auc=number(metric['auroc'] if 'auroc' in metric else metric['auc_probability'])
            ap=number(metric['average_precision'] if 'average_precision' in metric else metric['ap_probability'])
            tp,tn,fp,fn=(int(metric[k]) for k in ('tp','tn','fp','fn'))
            if metric['threshold']!=.5 or min(tp,tn,fp,fn)<0 or tp+tn+fp+fn!=5849 or tp+fn!=191:raise ValueError('Not full official valid metrics')
            r.update(metrics=dict(auroc=auc,average_precision=ap,fn=fn,fp=fp,threshold=metric['threshold']),report_sha256=sha(path),evidence='final_report_checked')
        if item.get('submission'):
            path=under(root,item['submission']['path'])
            if not path.is_file():r['submission']='missing'
            elif sha(path)!=item['submission']['sha256']:raise ValueError('Candidate CSV identity mismatch: '+item['id'])
            else:r['submission']=dict(sha256=sha(path),verified=True)
        result.append(r)
    return result


def markdown(rows):
    lines=['# 实验与候选结果总表','',
      '只汇总精选最终验证记录。平台分数来自用户反馈，未独立读取平台回执；缺失分数不按0处理。AUROC为0～1，平台分为百分制，不能当作同一份测试指标。','',
      '|候选|最终valid AUROC|AP|FN/FP @0.5|平台分（用户反馈）|状态|','|---|---:|---:|---|---:|---|']
    for r in rows:
        m=r['metrics'];auc=f"{m['auroc']:.6f}" if m else '—';ap=f"{m['average_precision']:.6f}" if m else '—'
        confusion=f"{m['fn']}/{m['fp']}" if m else '—'
        score=f"{r['platform']['score']:.2f}" if r['platform'] else '待反馈/未提交'
        state=r['decision'] if r['evidence']!='missing_report' else '本机缺少报告，未填指标'
        lines.append(f"|{r['title']}|{auc}|{ap}|{confusion}|{score}|{state}|")
    lines+=['','当前没有真实T4验收结果；4090D均值低于100ms不等于单图/T4达标。原图/翻转、R16/R27及选模/最终B1结果不作直接因果归因。未完成、缺失或门限失败的报告不能冒充最终结果。','', '## 结论与后续','']
    lines += [f"- {r['title']}：{r['reason']}。" for r in rows]
    lines+=['','后续只需更新平台反馈与完成T4验收；不自动挑选本地最高分、不搜索融合系数、不重新训练、不上传平台。生成目录的summary.json记录来源SHA及候选CSV核验状态；版本库中的本表为静态快照，私有摘要不入库。','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',required=True,type=Path)
    p.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1]/'experiments/results_catalog.json')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    catalog=json.loads(a.catalog.read_text(encoding='utf-8'));rows=collect(catalog,a.results_root)
    a.output.mkdir(parents=True,exist_ok=False)
    summary=dict(schema='experiment-summary-v1',catalog_sha256=sha(a.catalog),scope=catalog['scope'],entries=rows,automatic_model_selection=False)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.output/'summary.md').write_text(markdown(rows),encoding='utf-8')
    print('Summarized',len(rows),'entries; reports missing:',sum(r['evidence']=='missing_report' for r in rows))

if __name__=='__main__':main()
