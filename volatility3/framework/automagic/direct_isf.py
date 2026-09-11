# Direct ISF loader — bypasses Vol3's banner scan.
#
# Vol3 normally scans the entire physical image for a linux_banner string.
# On large USB-backed images this can take hours.  This automagic reads the
# ISF directly, probes init_task at the expected physical address (O(1)),
# and wires up the kernel requirements.
#
# Installation:
#   Copy into volatility3/framework/automagic/direct_isf.py
#
# Usage:
#   VOL3_ISF=/path/to/profile.json python vol.py -f image.vmem linux.pslist
#
# Environment variables:
#   VOL3_ISF        Path to dwarf2json / necromancer ISF .json file.
#   VOL3_PCI_HOLE   PCI MMIO hole size in hex/dec (e.g. 0x40000000).
#   VOL3_PHYS_BASE  Physical base address hint for KASLR (hex/dec).
#
# Example (VMware x86_64 guest with 1 GiB PCI hole):
#   VOL3_PCI_HOLE=0x40000000 VOL3_PHYS_BASE=0x15eb600000 \
#       VOL3_ISF=/path/to/profile.json \
#       python vol.py -f image.vmem linux.pslist

import json
import logging
import os
import struct
from collections import defaultdict
from typing import List

from volatility3.framework import constants, interfaces
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import intel

vollog = logging.getLogger(__name__)


def _virt_to_phys(addr: int) -> int:
    """x86_64 __START_KERNEL_map virtual -> physical (no KASLR)."""
    if addr > 0xFFFF_FFFF_8000_0000:
        return addr - 0xFFFF_FFFF_8000_0000
    return addr - 0xC000_0000


# Map dwarf2json / DWARF type names to Vol3 native table names.
_TYPE_MAP = {
    "long long unsigned int": "unsigned long long",
    "long long int": "long long",
    "long unsigned int": "unsigned long",
    "long int": "long",
    "short int": "short",
    "short unsigned int": "unsigned short int",
    "signed char": "char",
    "_Bool": "unsigned char",
    "ssizetype": "long",
}

# Symbol-name -> type annotation for necromancer ISFs (address-only symbols).
_SYMBOL_TYPES = {
    "init_task": {"kind": "struct", "name": "task_struct"},
    "init_files": {"kind": "struct", "name": "files_struct"},
    "init_mm": {"kind": "struct", "name": "mm_struct"},
    "init_pid_ns": {"kind": "struct", "name": "pid_namespace"},
    "init_net": {"kind": "struct", "name": "net"},
    "init_nsproxy": {"kind": "struct", "name": "nsproxy"},
    "init_uts_ns": {"kind": "struct", "name": "uts_namespace"},
    "modules": {"kind": "struct", "name": "list_head"},
    "runqueues": {"kind": "struct", "name": "rq"},
    "super_blocks": {"kind": "struct", "name": "list_head"},
}


class DirectISFAutomagic(interfaces.automagic.AutomagicInterface):
    """Set VOL3_ISF=/path/to/profile.json to skip the banner scan."""

    priority = 35  # after LayerStacker (10), before SymbolFinder (40)
    exclusion_list = ["mac", "windows"]

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return []

    def __call__(self, context, config_path, requirement, progress_callback=None):
        isf_path = os.environ.get("VOL3_ISF")
        if not isf_path or not os.path.isfile(isf_path):
            return

        # Find unsatisfied kernel requirements (layer + symbol table)
        reqs = self.find_requirements(
            context, config_path, requirement,
            (requirements.TranslationLayerRequirement,
             requirements.SymbolTableRequirement),
            shortcut=False,
        )

        by_parent = defaultdict(dict)
        for sub_path, req in reqs:
            parent = interfaces.configuration.parent_path(sub_path)
            if isinstance(req, requirements.TranslationLayerRequirement):
                by_parent[parent]["tl"] = (sub_path, req)
            elif isinstance(req, requirements.SymbolTableRequirement):
                by_parent[parent]["st"] = (sub_path, req)

        for parent, pair in by_parent.items():
            if "tl" not in pair or "st" not in pair:
                continue
            tl_path, tl_req = pair["tl"]
            st_path, st_req = pair["st"]
            if not st_req.unsatisfied(context, parent):
                continue

            if self._build_kernel(context, isf_path,
                                  parent, tl_path, st_path, tl_req, st_req):
                return

    # -----------------------------------------------------------------

    def _build_kernel(self, context, isf_path,
                      parent_path, tl_path, st_path, tl_req, st_req):
        vollog.info("DirectISF: loading %s", isf_path)

        # --- Ensure a physical layer exists ---
        phys_layer = self._get_or_create_phys_layer(context)
        if phys_layer is None:
            return False

        vollog.info("DirectISF: physical layer = %s", phys_layer.name)

        # --- Load & patch ISF ---
        with open(isf_path, "r") as f:
            isf = json.load(f)
        patched_path = self._patch_isf(isf)

        metadata = isf.get("metadata", {})
        symbols = isf.get("symbols", {})
        user_types = isf.get("user_types", {})

        # --- Locate init_task ---
        it_entry = symbols.get("init_task")
        if not it_entry:
            vollog.warning("DirectISF: ISF has no init_task symbol")
            return False
        it_va = it_entry["address"] if isinstance(it_entry, dict) else it_entry

        ts = user_types.get("task_struct", {}).get("fields", {})
        comm_off = ts.get("comm", {}).get("offset")
        if comm_off is None:
            vollog.warning("DirectISF: ISF has no task_struct.comm offset")
            return False

        found_pa, phys_base, kaslr_virt = self._find_init_task(
            phys_layer, it_va, comm_off, metadata,
        )
        if found_pa is None:
            return False

        vollog.info("DirectISF: init_task at PA %#x (phys_base=%#x)",
                    found_pa, phys_base)

        # --- Virtual ASLR shift ---
        # For compile-time ISFs (dwarf2json), the KASLR brute-force
        # already found the virtual offset.  For runtime ISFs
        # (necromancer), derive it from the init_task.files pointer.
        aslr_shift = kaslr_virt
        if not aslr_shift:
            files_off = ts.get("files", {}).get("offset")
            init_files = symbols.get("init_files")
            if files_off is not None and init_files is not None:
                ifva = (init_files["address"]
                        if isinstance(init_files, dict) else init_files)
                try:
                    ptr = struct.unpack(
                        "<Q",
                        phys_layer.read(found_pa + files_off, 8, pad=True),
                    )[0]
                    aslr_shift = ptr - ifva
                except Exception:
                    pass

        # --- DTB ---
        dtb_sym = None
        for name in ("init_top_pgt", "init_level4_pgt", "swapper_pg_dir"):
            if name in symbols:
                dtb_sym = name
                break
        if not dtb_sym:
            vollog.warning("DirectISF: no DTB symbol in ISF")
            return False

        dtb_entry = symbols[dtb_sym]
        dtb_va = dtb_entry["address"] if isinstance(dtb_entry, dict) else dtb_entry
        dtb = _virt_to_phys(dtb_va + aslr_shift) + phys_base

        vollog.info("DirectISF: aslr=%#x kaslr=%#x dtb=%#x",
                    aslr_shift, phys_base, dtb)

        # --- Build Intel layer ---
        join = interfaces.configuration.path_join
        layer_class = (intel.LinuxIntel32e
                       if dtb_sym in ("init_top_pgt", "init_level4_pgt")
                       else intel.LinuxIntel)

        layer_name = context.layers.free_layer_name("IntelLayer")
        cfg = join("DirectISFHelper", layer_name)
        context.config[join(cfg, "memory_layer")] = phys_layer.name
        context.config[join(cfg, "page_map_offset")] = dtb

        new_layer = layer_class(
            context, config_path=cfg, name=layer_name,
            metadata={"os": "Linux"},
        )
        new_layer.config["kernel_virtual_offset"] = aslr_shift
        context.layers.add_layer(new_layer)

        # Satisfy TranslationLayerRequirement and merge config so
        # KernelModule can find kernel_virtual_offset.
        context.config[tl_path] = layer_name
        context.config.merge(tl_path, new_layer.build_configuration())

        # --- Build symbol table ---
        isf_url = "file://" + os.path.abspath(patched_path)
        from volatility3.framework.symbols import (
            linux as linux_symbols, native,
        )
        table_name = context.symbol_space.free_table_name("DirectISF")
        table = linux_symbols.LinuxKernelIntermedSymbols(
            context, join(parent_path, "DirectISF_config"),
            name=table_name, isf_url=isf_url, validate=False,
            native_types=native.x64NativeTable,
        )
        context.symbol_space.append(table)

        # Satisfy SymbolTableRequirement
        st_name = st_path.split(
            interfaces.configuration.CONFIG_SEPARATOR
        )[-1]
        context.config[join(parent_path, st_name)] = table_name

        vollog.info("DirectISF: kernel ready (DTB=%#x, layer=%s)",
                    dtb, layer_name)
        return True

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_or_create_phys_layer(self, context):
        """Return the topmost non-Intel layer, creating FileLayer +
        PciHoleLayer from env vars if nothing exists yet."""
        for lname in reversed(list(context.layers)):
            layer = context.layers[lname]
            if not isinstance(layer, intel.Intel):
                return layer

        # No layers — create FileLayer from stacker config
        location = context.config.get(
            "automagic.LayerStacker.single_location", None
        )
        if not location:
            vollog.warning("DirectISF: no layers and no single_location")
            return None

        from volatility3.framework.layers import physical
        join = interfaces.configuration.path_join
        fl_name = context.layers.free_layer_name("FileLayer")
        fl_cfg = join("DirectISFHelper", fl_name)
        context.config[join(fl_cfg, "location")] = location
        phys_layer = physical.FileLayer(context, fl_cfg, fl_name)
        context.layers.add_layer(phys_layer)

        # Apply PCI hole if configured
        pci_size_str = os.environ.get("VOL3_PCI_HOLE")
        if pci_size_str:
            try:
                pci_size = int(pci_size_str, 0)
            except ValueError:
                pci_size = 0
            if pci_size:
                from volatility3.framework.layers.pci_hole import (
                    PciHoleLayer, PCI_START,
                )
                ph_name = context.layers.free_layer_name("PciHoleLayer")
                ph_cfg = join("DirectISFHelper", ph_name)
                context.config[join(ph_cfg, "base_layer")] = phys_layer.name
                context.config[join(ph_cfg, "pci_start")] = PCI_START
                context.config[join(ph_cfg, "pci_size")] = pci_size
                phys_layer = PciHoleLayer(context, ph_cfg, ph_name)
                context.layers.add_layer(phys_layer)
                vollog.info(
                    "DirectISF: PCI hole %#x at %#x", pci_size, PCI_START
                )

        return phys_layer

    @staticmethod
    def _patch_isf(isf):
        """Normalise an ISF dict for Vol3 compatibility and write a
        patched temp file.  Returns the temp file path."""
        import tempfile

        for key in ("enums", "base_types", "user_types", "symbols", "metadata"):
            isf.setdefault(key, {})

        isf["metadata"].setdefault("format", "6.2.0")

        # base_types: "size" -> "length" (Vol3 v1 format for native matching)
        for bt in isf["base_types"].values():
            if "size" in bt and "length" not in bt:
                bt["length"] = bt.pop("size")

        # Rename DWARF type names to Vol3 native names
        for old, new in _TYPE_MAP.items():
            if old in isf["base_types"]:
                if new not in isf["base_types"]:
                    isf["base_types"][new] = isf["base_types"][old]
                del isf["base_types"][old]

        def _fix_refs(obj):
            if isinstance(obj, dict):
                if obj.get("kind") == "base" and obj.get("name") in _TYPE_MAP:
                    obj["name"] = _TYPE_MAP[obj["name"]]
                for v in obj.values():
                    _fix_refs(v)
            elif isinstance(obj, list):
                for v in obj:
                    _fix_refs(v)

        _fix_refs(isf.get("user_types", {}))
        _fix_refs(isf.get("enums", {}))
        _fix_refs(isf.get("symbols", {}))

        # Add type annotations for address-only symbols (necromancer ISFs)
        for sym_name, sym_type in _SYMBOL_TYPES.items():
            sym = isf["symbols"].get(sym_name)
            if sym and isinstance(sym, dict) and "type" not in sym:
                if (sym_type["kind"] == "struct"
                        and sym_type["name"] in isf["user_types"]):
                    sym["type"] = sym_type

        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as tf:
            json.dump(isf, tf)
        return path

    @staticmethod
    def _find_init_task(phys_layer, it_va, comm_off, metadata):
        """Probe for init_task's 'swapper' comm string.  Returns
        ``(pa, phys_base, kaslr_virt)`` or ``(None, 0, 0)``."""
        env_pb = os.environ.get("VOL3_PHYS_BASE")
        if env_pb:
            try:
                phys_base = int(env_pb, 0)
            except ValueError:
                phys_base = 0
        else:
            phys_base = metadata.get("phys_base", 0)
        it_offset = _virt_to_phys(it_va)

        def _is_swapper(pa):
            try:
                data = phys_layer.read(pa + comm_off, 16, pad=True)
            except Exception:
                return False
            return (data.startswith(b"swapper/0\x00") or
                    (data.startswith(b"swapper\x00")
                     and data[8:16] == b"\x00" * 8))

        # Fast O(1) probe with known phys_base
        bases = list(dict.fromkeys([phys_base, 0]))
        for pb in bases:
            pa = it_offset + pb
            if _is_swapper(pa):
                return pa, pb, 0

        # Brute-force virtual KASLR offset (x86_64: 2 MiB-aligned,
        # up to ~1.5 GiB).  ~768 probes of 16 bytes each.
        KASLR_ALIGN = 0x200000   # 2 MiB
        KASLR_MAX = 0x60000000   # 1.5 GiB
        vollog.info(
            "DirectISF: scanning KASLR offsets (up to %d positions)...",
            KASLR_MAX // KASLR_ALIGN,
        )
        for pb in bases:
            for kv in range(KASLR_ALIGN, KASLR_MAX, KASLR_ALIGN):
                pa = it_offset + pb + kv
                if _is_swapper(pa):
                    vollog.info("DirectISF: KASLR virtual offset = %#x", kv)
                    return pa, pb, kv

        vollog.warning(
            "DirectISF: swapper not found (phys_base=%s)",
            [hex(b) for b in bases],
        )
        return None, 0, 0
