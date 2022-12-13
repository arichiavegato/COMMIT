import commit
from commit import trk2dictionary

trk2dictionary.run(
    filename_tractogram = 'demo01_fibers.tck',
    filename_peaks      = 'peaks.nii.gz',
    filename_mask       = 'WM.nii.gz',
    fiber_shift         = 0.5,
    peaks_use_affine    = True
)
