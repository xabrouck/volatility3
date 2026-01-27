# This file is Copyright 2024 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

import struct
from typing import List, Tuple

from volatility3.framework import interfaces, exceptions
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import linear


class AArch64(linear.LinearlyMappedLayer):
    """AArch64 translation layer supporting both 3-level and 4-level page tables.

    Translates virtual addresses to physical using the page global directory.
    Supports 4KB pages with either 39-bit VA (3-level) or 48-bit VA (4-level).
    """

    _direct_metadata = {
        "architecture": "AArch64",
        "mapped": True,
    }

    # AArch64 is little-endian and uses 64-bit entries
    _entry_format = "<Q"

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
            requirements.TranslationLayerRequirement(
                name="memory_layer", optional=False
            ),
            requirements.IntRequirement(name="page_map_offset", optional=False),
            requirements.IntRequirement(name="kernel_virtual_offset", optional=True),
            requirements.StringRequirement(name="kernel_banner", optional=True),
            requirements.IntRequirement(name="page_table_levels", optional=True),
        ]

    def __init__(self, context, config_path, name, metadata=None):
        super().__init__(
            context=context, config_path=config_path, name=name, metadata=metadata
        )
        self._base_layer = self.config["memory_layer"]
        self._pgd_addr = self.config["page_map_offset"]
        self._translation_cache = {}
        self._page_table_levels = self.config.get("page_table_levels", 4)

    @property
    def dependencies(self) -> List[str]:
        return [self._base_layer]

    @property
    def minimum_address(self) -> int:
        return 0

    @property
    def maximum_address(self) -> int:
        return (2**64) - 1

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

        Supports both 3-level (39-bit VA) and 4-level (48-bit VA) page tables.
        """
        # Check cache first
        page_vaddr = vaddr & ~0xFFF
        if page_vaddr in self._translation_cache:
            page_phys, page_size, pte_entry = self._translation_cache[page_vaddr]
            return page_phys + (vaddr & 0xFFF), page_size, pte_entry

        if self._page_table_levels == 3:
            return self._translate_3level(vaddr)
        else:
            return self._translate_4level(vaddr)

    def _translate_3level(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate using 3-level page table (39-bit VA)."""
        page_vaddr = vaddr & ~0xFFF

        # Extract indices from virtual address (lower 39 bits)
        va_bits = vaddr & 0x7FFFFFFFFF
        pgd_idx = (va_bits >> 30) & 0x1FF
        pmd_idx = (va_bits >> 21) & 0x1FF
        pte_idx = (va_bits >> 12) & 0x1FF

        # Level 0: PGD (points to PMD)
        pgd_entry_addr = self._pgd_addr + pgd_idx * 8
        pgd_entry = self._read_phys_u64(pgd_entry_addr)
        if not (pgd_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PGD entry"
            )

        # Check for 1GB block at PGD level
        if not (pgd_entry & self._PTE_TABLE):
            block_addr = pgd_entry & 0xFFFFC0000000
            return block_addr + (va_bits & 0x3FFFFFFF), 1 << 30, pgd_entry

        # Level 1: PMD
        pmd_addr = pgd_entry & self._PTE_ADDR_MASK
        pmd_entry = self._read_phys_u64(pmd_addr + pmd_idx * 8)
        if not (pmd_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PMD entry"
            )

        # Check for 2MB block
        if not (pmd_entry & self._PTE_TABLE):
            block_addr = pmd_entry & 0xFFFFFFE00000
            return block_addr + (va_bits & 0x1FFFFF), 1 << 21, pmd_entry

        # Level 2: PTE
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

    def _translate_4level(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate using 4-level page table (48-bit VA)."""
        page_vaddr = vaddr & ~0xFFF

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

    def canonicalize(self, addr: int) -> int:
        """Canonicalizes an address by sign-extending from the VA width.

        AArch64 kernel addresses have upper bits set (TTBR1 space).
        For 48-bit VA, addresses >= 0xFFFF000000000000 are kernel space.
        For 39-bit VA, addresses >= 0xFFFFFF8000000000 are kernel space.
        """
        if self._page_table_levels == 3:
            # 39-bit VA: sign extend from bit 38
            if addr & (1 << 38):
                return addr | 0xFFFFFF8000000000
        else:
            # 48-bit VA: sign extend from bit 47
            if addr & (1 << 47):
                return addr | 0xFFFF000000000000
        return addr


class LinuxAArch64(AArch64):
    """Linux-specific AArch64 layer."""
