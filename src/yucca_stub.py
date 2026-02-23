"""
Minimal Yucca stub for mmunetvae (used only in predict(); training uses forward() only).
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_stub_dir = _root / "src" / "yucca_stub_pkg"
_stub_dir.mkdir(parents=True, exist_ok=True)

# Create yucca package structure
for subdir in ["modules", "modules/networks", "modules/networks/utils"]:
    ( _stub_dir / "yucca" / subdir).mkdir(parents=True, exist_ok=True)

def _get_steps_for_sliding_window(shape, patch_size, overlap):
    """Simple sliding window steps (from nnUNet style)."""
    steps = []
    for i in range(len(shape)):
        step = max(1, int(patch_size[i] * (1 - overlap)))
        max_start = max(0, shape[i] - patch_size[i])
        st = list(range(0, max_start + 1, step))
        if st and st[-1] != max_start:
            st.append(max_start)
        steps.append(st)
    return steps

# Write the stub module
_stub_code = f'''def get_steps_for_sliding_window(shape, patch_size, overlap):
    """Stub implementation for sliding window."""
    steps = []
    for i in range(len(shape)):
        step = max(1, int(patch_size[i] * (1 - overlap)))
        max_start = max(0, shape[i] - patch_size[i])
        st = list(range(0, max_start + 1, step))
        if st and st[-1] != max_start:
            st.append(max_start)
        steps.append(st)
    return steps
'''
(_stub_dir / "yucca" / "__init__.py").write_text("")
(_stub_dir / "yucca" / "modules" / "__init__.py").write_text("")
(_stub_dir / "yucca" / "modules" / "networks" / "__init__.py").write_text("")
(_stub_dir / "yucca" / "modules" / "networks" / "utils" / "__init__.py").write_text("")
(_stub_dir / "yucca" / "modules" / "networks" / "utils" / "get_steps_for_sliding_window.py").write_text(_stub_code)

# Patch get_steps_for_sliding_window into sys.modules if yucca not present
if "yucca.modules.networks.utils.get_steps_for_sliding_window" not in sys.modules:
    sys.path.insert(0, str(_stub_dir))
