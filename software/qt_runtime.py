"""Use the bundled Qt plugins through Unicode paths on Windows."""
from pathlib import Path
import os

from PyQt5 import QtCore


def configure_qt():
    # PyQt5's generated qt.conf can lose non-ASCII characters in a venv prefix.
    # Pass the actual Unicode path directly instead of relying on that prefix.
    plugins = Path(QtCore.__file__).resolve().parent/'Qt5'/'plugins'
    if plugins.is_dir():
        QtCore.QCoreApplication.setLibraryPaths([str(plugins)])
        os.environ['QT_PLUGIN_PATH'] = str(plugins)
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins/'platforms')
    if os.name == 'nt' and os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        fonts = Path(os.environ.get('SystemRoot', 'C:/Windows'))/'Fonts'
        if fonts.is_dir():
            os.environ.setdefault('QT_QPA_FONTDIR', str(fonts))
