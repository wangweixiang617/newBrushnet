"""Reference metrics in [0,1]. Empty regions -> NaN, never a fabricated perfect score."""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, sobel
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity


def masked_mean(values,mask):
    mask=np.asarray(mask,dtype=np.float64)
    if values.ndim==3:
        values=np.mean(values,axis=-1)
    return float(np.sum(values*mask)/mask.sum()) if mask.sum()>0 else float('nan')


def regions(hole,radii=(3,5,9)):
    m=(hole>0.5).astype(np.float32)
    out={'full':np.ones_like(m),'hole':m,'known':1-m}
    for r in radii:
        if r<1:
            raise ValueError('Boundary radius must be positive')
        dil=maximum_filter(m,size=2*r+1,mode='constant',cval=0)
        ero=minimum_filter(m,size=2*r+1,mode='constant',cval=1)
        out[f'boundary{r}']=dil-ero
    return out


def haar_np(x):
    a,b,c,d=x[::2,::2],x[::2,1::2],x[1::2,::2],x[1::2,1::2]
    return (a+b+c+d)/2,((-a-b+c+d)/2,(-a+b-c+d)/2,(a-b-c+d)/2)


def reference_metrics(pred,gt,hole,radii=(3,5,9)):
    pred,gt=np.asarray(pred,dtype=np.float64),np.asarray(gt,dtype=np.float64)
    if pred.shape!=gt.shape or pred.ndim!=3 or pred.shape[-1]!=3 or hole.shape!=pred.shape[:2]:
        raise ValueError('Expected matched HWC RGB images and HW mask')
    if pred.min()<0 or pred.max()>1 or gt.min()<0 or gt.max()>1:
        raise ValueError('Metric inputs must be [0,1]')
    masks=regions(hole,radii)
    square=(pred-gt)**2
    lab_error=np.linalg.norm(rgb2lab(pred)-rgb2lab(gt),axis=-1)
    lum=lambda x: x@np.array([.2126,.7152,.0722])
    yp,yg=lum(pred),lum(gt)
    grad=np.sqrt(sum((sobel(yp,axis=k,mode='reflect')/8-sobel(yg,axis=k,mode='reflect')/8)**2 for k in (0,1)))
    _,ssim=structural_similarity(gt,pred,data_range=1.,channel_axis=-1,gaussian_weights=True,sigma=1.5,
                                use_sample_covariance=False,full=True)
    # Exclude the 5-pixel outer border from SSIM aggregation, as in the standard valid window.
    valid=np.zeros(hole.shape); valid[5:-5,5:-5]=1
    wave=np.zeros(hole.shape)
    lp,lg=pred,gt
    for j in range(3):
        lp,hp=haar_np(lp); lg,hg=haar_np(lg)
        error=sum(np.mean(np.abs(a-b),axis=-1) for a,b in zip(hp,hg))/3
        # Per-level amplitude scale compensation; diagnostic only, not an artifact detector.
        error=error/(2**(j+1))
        wave+=np.repeat(np.repeat(error,2**(j+1),axis=0),2**(j+1),axis=1)/3
    out={}
    for name,m in masks.items():
        mse=masked_mean(square,m)
        out[name+'_mse']=mse
        out[name+'_psnr']=float('nan') if np.isnan(mse) else float('inf') if mse==0 else -10*np.log10(mse)
        out[name+'_ssim']=masked_mean(ssim,m*valid)
        out[name+'_deltaE76']=masked_mean(lab_error,m)
        out[name+'_gradient']=masked_mean(grad,m)
        out[name+'_wavelet_l1']=masked_mean(wave,m)
    # Match user's old RGB-sum / spatial-pixel denominator for historical comparisons.
    old=3*out['known_mse']
    out['legacy_known_mse']=old
    out['legacy_known_psnr']=1000. if old<1e-10 else -10*np.log10(old)
    out['hole_fraction']=float(hole.mean())
    return out
