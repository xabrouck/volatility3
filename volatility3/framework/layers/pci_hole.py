# PCI hole adapter for VMware x86_64 .vmem files.
#
# On many VMware x86_64 guests the BIOS reserves a PCI MMIO aperture
# (typically 1 GiB at GPA 0xC000_0000..0xFFFF_FFFF).  The .vmem file
# omits this gap, so every guest-physical address above 4 GiB is shifted
# down by the aperture size in the file.
#
# Installation:
#   Copy this file into volatility3/framework/layers/pci_hole.py
#   (auto-discovered — no other changes needed).
#
# Usage:
#   VOL3_PCI_HOLE=0x40000000 python vol.py -f image.vmem linux.pslist
#
# Also auto-detected from ISF metadata.pci_hole if present.

import logging
import os
import re
from typing import Any, Dict, List, Optional

from volatility3.framework import constants, interfaces
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import physical, segmented

vollog = logging.getLogger(__name__)

PCI_START = 0xC000_0000  # standard x86 PCI MMIO window start


class PciHoleLayer(segmented.SegmentedLayer):
    """Remaps guest-physical addresses to file offsets by inserting a
    PCI MMIO gap.

    Segment layout (for a 1 GiB hole at 3 GiB):
        GPA [0, 0xC000_0000)              -> file [0, 0xC000_0000)
        GPA [0xC000_0000, 0x1_0000_0000)  -> unmapped (PCI aperture)
        GPA [0x1_0000_0000, ...)           -> file [0xC000_0000, ...)
    """

    def __init__(
        self,
        context: interfaces.context.ContextInterface,
        config_path: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._context = context
        self._config_path = config_path
        self._base_layer = self.config["base_layer"]
        self._pci_start = self.config.get("pci_start", PCI_START)
        self._pci_size = self.config["pci_size"]
        super().__init__(
            context=context, config_path=config_path, name=name, metadata=metadata
        )

    def _load_segments(self) -> None:
        base = self._context.layers[self._base_layer]
        file_size = base.maximum_address + 1

        # Segment 1: before the hole (identity-mapped)
        seg1_len = min(self._pci_start, file_size)
        if seg1_len > 0:
            self._segments.append((0, 0, seg1_len, seg1_len))

        # Segment 2: after the hole (shifted up by pci_size in GPA space)
        remaining = file_size - self._pci_start
        if remaining > 0:
            gpa_after_hole = self._pci_start + self._pci_size
            self._segments.append(
                (gpa_after_hole, self._pci_start, remaining, remaining)
            )

    @property
    def dependencies(self) -> List[str]:
        return [self._base_layer]

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.TranslationLayerRequirement(
                name="base_layer", optional=False
            ),
            requirements.IntRequirement(
                name="pci_start", optional=True, default=PCI_START
            ),
            requirements.IntRequirement(name="pci_size", optional=False),
        ]


class PciHoleStacker(interfaces.automagic.StackerLayerInterface):
    """Auto-insert a PCI hole adapter when an ISF profile or environment
    variable indicates one is needed."""

    stack_order = 25  # after FileLayer (~10), before LinuxIntelStacker (35)

    @classmethod
    def stack(
        cls,
        context: interfaces.context.ContextInterface,
        layer_name: str,
        progress_callback: constants.ProgressCallback = None,
    ) -> Optional[interfaces.layers.DataLayerInterface]:
        layer = context.layers[layer_name]
        if not isinstance(layer, physical.FileLayer):
            return None

        pci_size = cls._detect_pci_hole()
        if not pci_size:
            return None

        vollog.info(
            "Applying PCI hole: %#x bytes at GPA %#x", pci_size, PCI_START
        )

        join = interfaces.configuration.path_join
        new_name = context.layers.free_layer_name("PciHoleLayer")
        cfg = join("automagic", "layer_stacker", "stack", new_name)
        context.config[join(cfg, "base_layer")] = layer_name
        context.config[join(cfg, "pci_start")] = PCI_START
        context.config[join(cfg, "pci_size")] = pci_size
        return PciHoleLayer(context, cfg, new_name)

    @classmethod
    def _detect_pci_hole(cls) -> int:
        # 1. Explicit override via environment variable
        env = os.environ.get("VOL3_PCI_HOLE")
        if env:
            try:
                return int(env, 0)
            except ValueError:
                vollog.warning("VOL3_PCI_HOLE=%r is not a valid integer", env)

        # 2. Scan ISF files in the configured symbols paths for pci_hole metadata.
        #    At this stacking stage symbol tables haven't been loaded yet, so we
        #    peek at the raw JSON of every .json file in the symbols directories.
        try:
            import volatility3.symbols
            for symbols_dir in volatility3.symbols.__path__:
                hole = cls._scan_isf_dir(symbols_dir)
                if hole:
                    return hole
        except Exception:
            pass

        return 0

    @classmethod
    def _scan_isf_dir(cls, path: str) -> int:
        """Walk *path* looking for ISF .json files with metadata.pci_hole."""
        if not os.path.isdir(path):
            # -s might point at a single file; try it directly
            if os.path.isfile(path):
                return cls._read_pci_hole(path)
            return 0
        for root, _dirs, files in os.walk(path):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                hole = cls._read_pci_hole(os.path.join(root, fname))
                if hole:
                    return hole
        return 0

    @staticmethod
    def _read_pci_hole(filepath: str) -> int:
        """Return metadata.pci_hole from an ISF JSON, or 0."""
        try:
            with open(filepath, "rb") as f:
                head = f.read(8192)
            m = re.search(rb'"pci_hole"\s*:\s*(\d+)', head)
            return int(m.group(1)) if m else 0
        except Exception:
            return 0
