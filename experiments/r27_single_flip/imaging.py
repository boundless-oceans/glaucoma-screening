"""Mirror oriented draft-decoded pixels, without JPEG re-encoding or ROI reuse."""
from PIL import Image, ImageOps


def prepare_view(path,mirrored=True,size=512):
    # Keep the archived _prepare geometry, original aspect ratio and rounding.
    # The only extra operation is mirror AFTER EXIF orientation and RGB decode.
    from glaucoma.data import fundus_bbox
    with Image.open(path) as im:
        width,height=im.size
        if im.getexif().get(274,1) in (5,6,7,8):width,height=height,width
        im.draft('RGB',(768,768))
        image=ImageOps.exif_transpose(im) if im.getexif().get(274,1)!=1 else im
        if image.mode!='RGB':image=image.convert('RGB')
        else:image.load()
    if mirrored:image=ImageOps.mirror(image)
    dw,dh=image.size
    x0,y0,x1,y1=fundus_bbox(image)
    actual_width=(x1-x0)*width/dw;actual_height=(y1-y0)*height/dh
    target=(size,max(1,round(actual_height*size/actual_width))) if actual_width>=actual_height else (max(1,round(actual_width*size/actual_height)),size)
    resized=image.crop((x0,y0,x1,y1)).resize(target,Image.Resampling.BICUBIC)
    full=ImageOps.pad(resized,(size,size),method=Image.Resampling.BICUBIC,color=(0,0,0))
    geometry=dict(size=size,field=(x0/dw,y0/dh,x1/dw,y1/dh),target=target,
                  padding=(round((size-target[0])/2),round((size-target[1])/2)))
    return full,geometry,image
