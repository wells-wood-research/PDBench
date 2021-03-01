import subprocess
label = subprocess.check_output(['git', 'rev-parse','--short', 'HEAD']).strip().decode('ascii')
__version__ = f"0.1_{label}"