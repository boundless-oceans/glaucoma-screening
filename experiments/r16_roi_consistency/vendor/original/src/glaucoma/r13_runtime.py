"""Keep deterministic dev anchors in the same runtime as final official inference."""
def require_official_dev_environment(torch_version,vision_version):
    if (str(torch_version),str(vision_version))!=('2.10.0+cu128','0.25.0+cu128'):
        raise ValueError('Dev ROI assets require the pinned official 2.10/cu128 runtime')
