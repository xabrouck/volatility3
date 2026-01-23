# This file is Copyright 2024 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

import struct
from typing import List, Tuple

from volatility3.framework import interfaces, exceptions
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import linear


class AArch64(linear.LinearlyMappedLayer):
    """AArch64 translation layer using 4-level page table walking.
    
    Translates virtual addresses to physical using the page global directory.
    Supports 4KB pages with 48-bit virtual addresses.
    """
    
    _direct_metadata = {
        "architecture": "AArch64",
        "mapped": True,
    }
    
    # Page table constants for 4KB pages
    _PAGE_SHIFT = 12
    _PAGE_SIZE = 1 << _PAGE_SHIFT  # 4096
    _PTE_VALID = 0x1
    _PTE_TABLE = 0x2  # For non-leaf entries, indicates table descriptor
    _PTE_DIRTY = 1 << 55  # DBM (Dirty Bit Modifier) - software managed
    _PTE_ADDR_MASK = 0x0000FFFFFFFFF000  # Physical address bits [47:12]
    
    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.TranslationLayerRequirement(name="memory_layer", optional=False),
            requirements.IntRequirement(name="page_map_offset", optional=False),
            requirements.IntRequirement(name="kernel_virtual_offset", optional=True),
            requirements.StringRequirement(name="kernel_banner", optional=True),
        ]
    
    def __init__(self, context, config_path, name, metadata=None):
        super().__init__(context=context, config_path=config_path, name=name, metadata=metadata)
        self._base_layer = self.config["memory_layer"]
        self._pgd_addr = self.config["page_map_offset"]
        self._translation_cache = {}
    
    @property
    def dependencies(self) -> List[str]:
        return [self._base_layer]
    
    @property
    def minimum_address(self) -> int:
        return 0
    
    @property
    def maximum_address(self) -> int:
        return (2 ** 64) - 1
    
    @property
    def page_shift(self) -> int:
        return self._PAGE_SHIFT
    
    @property
    def page_size(self) -> int:
        return self._PAGE_SIZE
    
    def _read_phys_u64(self, phys_addr: int) -> int:
        """Read a 64-bit value from physical memory."""
        try:
            data = self.context.layers[self._base_layer].read(phys_addr, 8)
            return struct.unpack("<Q", data)[0]
        except exceptions.InvalidAddressException:
            return 0
    
    def _translate_entry(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate virtual address and return (physical_addr, page_size, pte_entry).
        
        For 4KB pages with 48-bit VA:
        - Bits [47:39] = Level 0 index (PGD)
        - Bits [38:30] = Level 1 index (PUD)
        - Bits [29:21] = Level 2 index (PMD)
        - Bits [20:12] = Level 3 index (PTE)
        - Bits [11:0]  = Page offset
        """
        # Check cache first
        page_vaddr = vaddr & ~0xFFF
        if page_vaddr in self._translation_cache:
            page_phys, page_size, pte_entry = self._translation_cache[page_vaddr]
            return page_phys + (vaddr & 0xFFF), page_size, pte_entry
        
        # Extract indices from virtual address (lower 48 bits)
        va_bits = vaddr & 0xFFFFFFFFFFFF
        pgd_idx = (va_bits >> 39) & 0x1FF
        pud_idx = (va_bits >> 30) & 0x1FF
        pmd_idx = (va_bits >> 21) & 0x1FF
        pte_idx = (va_bits >> 12) & 0x1FF
        
        # Level 0: PGD
        pgd_entry_addr = self._pgd_addr + pgd_idx * 8
        pgd_entry = self._read_phys_u64(pgd_entry_addr)
        if not (pgd_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PGD entry"
            )
        
        # Level 1: PUD
        pud_addr = pgd_entry & self._PTE_ADDR_MASK
        pud_entry = self._read_phys_u64(pud_addr + pud_idx * 8)
        if not (pud_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PUD entry"
            )
        # Check for 1GB block
        if not (pud_entry & self._PTE_TABLE):
            block_addr = pud_entry & 0xFFFFC0000000
            return block_addr + (va_bits & 0x3FFFFFFF), 1 << 30, pud_entry
        
        # Level 2: PMD
        pmd_addr = pud_entry & self._PTE_ADDR_MASK
        pmd_entry = self._read_phys_u64(pmd_addr + pmd_idx * 8)
        if not (pmd_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PMD entry"
            )
        # Check for 2MB block
        if not (pmd_entry & self._PTE_TABLE):
            block_addr = pmd_entry & 0xFFFFFFE00000
            return block_addr + (va_bits & 0x1FFFFF), 1 << 21, pmd_entry
        
        # Level 3: PTE
        pte_addr = pmd_entry & self._PTE_ADDR_MASK
        pte_entry = self._read_phys_u64(pte_addr + pte_idx * 8)
        if not (pte_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PTE entry"
            )
        
        page_phys = pte_entry & self._PTE_ADDR_MASK
        page_size = 1 << 12  # 4KB
        
        # Cache the translation
        self._translation_cache[page_vaddr] = (page_phys, page_size, pte_entry)
        
        return page_phys + (vaddr & 0xFFF), page_size, pte_entry
    
    def _translate(self, offset: int) -> Tuple[int, int, str]:
        """Translate virtual to physical address."""
        phys, page_size, _ = self._translate_entry(offset)
        return phys, page_size, self._base_layer
    
    def is_valid(self, offset: int, length: int = 1) -> bool:
        try:
            physical, page_size, layer = self._translate(offset)
            return self.context.layers[layer].is_valid(physical, length)
        except exceptions.InvalidAddressException:
            return False
    
    def is_dirty(self, offset: int) -> bool:
        """Returns whether the page at offset is marked dirty."""
        try:
            _, _, pte_entry = self._translate_entry(offset)
            return bool(pte_entry & self._PTE_DIRTY)
        except exceptions.InvalidAddressException:
            return False
    
    def mapping(self, offset: int, length: int, ignore_errors: bool = False):
        try:
            physical, page_size, layer = self._translate(offset)
            yield offset, length, physical, length, layer
        except exceptions.InvalidAddressException:
            if not ignore_errors:
                raise


class LinuxAArch64(AArch64):
    """Linux-specific AArch64 layer."""
    pass
