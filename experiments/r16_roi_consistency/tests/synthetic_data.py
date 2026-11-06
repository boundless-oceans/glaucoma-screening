"""Artificial CPU fixtures, not competition rows; no image files are required."""
import csv
from pathlib import Path


def write_official(root,split):
    n,positive={'train':(51396,1657),'valid':(5849,191)}[split]
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/(split+'.csv')).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Image_name','Age','True_Label'])
        writer.writerows((f'{split}_{i:05d}.jpg','60.0',int(i>=n-positive)) for i in range(n))
