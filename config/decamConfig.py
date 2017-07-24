# use "instcal" exposures from the community pipeline until DecamIsrTask is up to snuff
from lsst.obs.decam.decamNullIsr import DecamNullIsrTask
config.isr.retarget(DecamNullIsrTask)

from lsst.pipe.tasks.setConfigFromEups import setPhotocalConfigFromEups
from lsst.meas.extensions.astrometryNet import LoadAstrometryNetObjectsTask
config.calibrate.astromRefObjLoader.retarget(LoadAstrometryNetObjectsTask)
config.calibrate.photoRefObjLoader.retarget(LoadAstrometryNetObjectsTask)

setPhotocalConfigFromEups(config.calibrate.photoCal)
