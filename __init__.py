from qgis.core import QgsApplication
from .processing_provider import Rain2FloodProvider

class Rain2FloodPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        
    def initGui(self):
        # Initialize and register the processing provider
        self.provider = Rain2FloodProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)
        
    def unload(self):
        # Unregister the processing provider
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)

def classFactory(iface):
    return Rain2FloodPlugin(iface)