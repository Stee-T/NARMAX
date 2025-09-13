# ################################################### Sub-namespaces ###################################################
# This section creates nested name spaces in the form NARMAX.NS."Something"
from . import Tools
from . import CTors # Expose all constuctors and helper functions

# ################################################## NARMAX namespace ##################################################
# Stuff injected directly into the NARMAX namespace: NARMAX."Something"

# Rest of the Lib: "from ."" is same Folder and ".Name" is subfolder import
from .Classes.NonLinearity import NonLinearity # import here to give acces to the user (add to namespaec rFOrLSR)
from .Classes.SymbolicOscillator_0_3 import SymbolicOscillator, Device
from .Classes.Arborescence import Arborescence
from .HelperFuncs import CutY
from .Validation import InitAndComputeBuffer, DefaultValidation

# Variables
Identity = NonLinearity( "id", lambda x: x ) # pre-define object for user