# This file is Copyright 2024 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

import struct
from typing import List, Tuple

from volatility3.framework import interfaces, exceptions
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import linear


class AArch64(linear.LinearlyMappedLayer):
    """AArch64 translation layer supporting multiple page sizes and VA widths.

    Translates virtual addresses to physical using the page global directory.
    Supports:
    - 4KB pages with 39-bit VA (3-level) or 48-bit VA (4-level)
    - 16KB pages with 47-bit VA (3-level) or 48-bit VA (4-level)
    """

    _direct_metadata = {
        "architecture": "AArch64",
        "mapped": True,
    }

    # AArch64 is little-endian and uses 64-bit entries/registers
    _entry_format = "<Q"
    _bits_per_register = 64

    # Common page table constants
    _PTE_VALID = 0x1
    _PTE_TABLE = 0x2  # For non-leaf entries, indicates table descriptor
    _PTE_DIRTY = 1 << 55  # DBM (Dirty Bit Modifier) - software managed

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
            requirements.IntRequirement(name="page_size_kb", optional=True),
        ]

    def __init__(self, context, config_path, name, metadata=None):
        super().__init__(
            context=context, config_path=config_path, name=name, metadata=metadata
        )
        self._base_layer = self.config["memory_layer"]
        self._pgd_addr = self.config["page_map_offset"]
        self._translation_cache = {}
        self._page_table_levels = self.config.get("page_table_levels", 4)

        # Page size configuration (default 4KB)
        page_size_kb = self.config.get("page_size_kb", 4)
        if page_size_kb == 16:
            self._PAGE_SHIFT = 14
            self._PAGE_SIZE = 1 << 14  # 16384
            self._PTE_ADDR_MASK = 0x0000FFFFFFFFC000  # Physical address bits [47:14]
            self._PTRS_PER_TABLE = 2048  # 11 bits index
            self._TABLE_SHIFT = 11
        else:
            # Default 4KB pages
            self._PAGE_SHIFT = 12
            self._PAGE_SIZE = 1 << 12  # 4096
            self._PTE_ADDR_MASK = 0x0000FFFFFFFFF000  # Physical address bits [47:12]
            self._PTRS_PER_TABLE = 512  # 9 bits index
            self._TABLE_SHIFT = 9

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

    @property
    def bits_per_register(self) -> int:
        return self._bits_per_register

    def _read_phys_u64(self, phys_addr: int) -> int:
        """Read a 64-bit value from physical memory."""
        try:
            data = self.context.layers[self._base_layer].read(phys_addr, 8)
            return struct.unpack("<Q", data)[0]
        except exceptions.InvalidAddressException:
            return 0

    def _translate_entry(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate virtual address and return (physical_addr, page_size, pte_entry).

        Supports multiple page sizes and VA widths:
        - 4KB pages: 39-bit VA (3-level) or 48-bit VA (4-level)
        - 16KB pages: 47-bit VA (3-level) or 48-bit VA (4-level)
        """
        # Check cache first
        page_mask = self._PAGE_SIZE - 1
        page_vaddr = vaddr & ~page_mask
        if page_vaddr in self._translation_cache:
            page_phys, page_size, pte_entry = self._translation_cache[page_vaddr]
            return page_phys + (vaddr & page_mask), page_size, pte_entry

        if self._page_table_levels == 3:
            return self._translate_3level(vaddr)
        else:
            return self._translate_4level(vaddr)

    def _translate_3level(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate using 3-level page table.

        For 4KB pages (39-bit VA):
          PGD[38:30] -> PMD[29:21] -> PTE[20:12] -> offset[11:0]
        For 16KB pages (47-bit VA):
          PGD[46:36] -> PMD[35:25] -> PTE[24:14] -> offset[13:0]
        """
        page_mask = self._PAGE_SIZE - 1
        page_vaddr = vaddr & ~page_mask
        idx_mask = self._PTRS_PER_TABLE - 1

        if self._PAGE_SHIFT == 14:
            # 16KB pages, 47-bit VA
            va_bits = vaddr & 0x7FFFFFFFFFFF  # Lower 47 bits
            pgd_idx = (va_bits >> 36) & idx_mask
            pmd_idx = (va_bits >> 25) & idx_mask
            pte_idx = (va_bits >> 14) & idx_mask
            # Block sizes for 16KB: PGD=64GB (not used), PMD=32MB
            pmd_block_mask = (1 << 25) - 1
            pmd_block_addr_mask = ~pmd_block_mask & 0x0000FFFFFFFFFFFF
        else:
            # 4KB pages, 39-bit VA
            va_bits = vaddr & 0x7FFFFFFFFF  # Lower 39 bits
            pgd_idx = (va_bits >> 30) & idx_mask
            pmd_idx = (va_bits >> 21) & idx_mask
            pte_idx = (va_bits >> 12) & idx_mask
            # Block sizes for 4KB: PGD=1GB, PMD=2MB
            pmd_block_mask = (1 << 21) - 1
            pmd_block_addr_mask = 0xFFFFFFE00000

        # Level 0: PGD (points to PMD)
        pgd_entry_addr = self._pgd_addr + pgd_idx * 8
        pgd_entry = self._read_phys_u64(pgd_entry_addr)
        if not (pgd_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PGD entry"
            )

        # Check for block at PGD level (1GB for 4KB pages, not typically used for 16KB)
        if not (pgd_entry & self._PTE_TABLE):
            if self._PAGE_SHIFT == 12:
                block_addr = pgd_entry & 0xFFFFC0000000
                return block_addr + (va_bits & 0x3FFFFFFF), 1 << 30, pgd_entry
            else:
                raise exceptions.InvalidAddressException(
                    self.name, vaddr, "Unexpected block descriptor at PGD level"
                )

        # Level 1: PMD
        pmd_addr = pgd_entry & self._PTE_ADDR_MASK
        pmd_entry = self._read_phys_u64(pmd_addr + pmd_idx * 8)
        if not (pmd_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PMD entry"
            )

        # Check for block at PMD level (2MB for 4KB, 32MB for 16KB)
        if not (pmd_entry & self._PTE_TABLE):
            block_addr = pmd_entry & pmd_block_addr_mask
            block_size = pmd_block_mask + 1
            return block_addr + (va_bits & pmd_block_mask), block_size, pmd_entry

        # Level 2: PTE
        pte_addr = pmd_entry & self._PTE_ADDR_MASK
        pte_entry = self._read_phys_u64(pte_addr + pte_idx * 8)
        if not (pte_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PTE entry"
            )

        page_phys = pte_entry & self._PTE_ADDR_MASK

        # Cache the translation
        self._translation_cache[page_vaddr] = (page_phys, self._PAGE_SIZE, pte_entry)

        return page_phys + (vaddr & page_mask), self._PAGE_SIZE, pte_entry

    def _translate_4level(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate using 4-level page table.

        For 4KB pages (48-bit VA):
          PGD[47:39] -> PUD[38:30] -> PMD[29:21] -> PTE[20:12] -> offset[11:0]
        For 16KB pages (48-bit VA):
          PGD[47:47] -> PUD[46:36] -> PMD[35:25] -> PTE[24:14] -> offset[13:0]
        """
        page_mask = self._PAGE_SIZE - 1
        page_vaddr = vaddr & ~page_mask
        idx_mask = self._PTRS_PER_TABLE - 1

        if self._PAGE_SHIFT == 14:
            # 16KB pages, 48-bit VA (4-level)
            va_bits = vaddr & 0xFFFFFFFFFFFF  # Lower 48 bits
            pgd_idx = (va_bits >> 47) & 0x1  # Only 1 bit for PGD in 48-bit/16KB
            pud_idx = (va_bits >> 36) & idx_mask
            pmd_idx = (va_bits >> 25) & idx_mask
            pte_idx = (va_bits >> 14) & idx_mask
            pmd_block_mask = (1 << 25) - 1
            pmd_block_addr_mask = ~pmd_block_mask & 0x0000FFFFFFFFFFFF
            pud_block_mask = (1 << 36) - 1
            pud_block_addr_mask = ~pud_block_mask & 0x0000FFFFFFFFFFFF
        else:
            # 4KB pages, 48-bit VA
            va_bits = vaddr & 0xFFFFFFFFFFFF  # Lower 48 bits
            pgd_idx = (va_bits >> 39) & idx_mask
            pud_idx = (va_bits >> 30) & idx_mask
            pmd_idx = (va_bits >> 21) & idx_mask
            pte_idx = (va_bits >> 12) & idx_mask
            pmd_block_mask = (1 << 21) - 1
            pmd_block_addr_mask = 0xFFFFFFE00000
            pud_block_mask = (1 << 30) - 1
            pud_block_addr_mask = 0xFFFFC0000000

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
        # Check for block at PUD level
        if not (pud_entry & self._PTE_TABLE):
            block_addr = pud_entry & pud_block_addr_mask
            block_size = pud_block_mask + 1
            return block_addr + (va_bits & pud_block_mask), block_size, pud_entry

        # Level 2: PMD
        pmd_addr = pud_entry & self._PTE_ADDR_MASK
        pmd_entry = self._read_phys_u64(pmd_addr + pmd_idx * 8)
        if not (pmd_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PMD entry"
            )
        # Check for block at PMD level
        if not (pmd_entry & self._PTE_TABLE):
            block_addr = pmd_entry & pmd_block_addr_mask
            block_size = pmd_block_mask + 1
            return block_addr + (va_bits & pmd_block_mask), block_size, pmd_entry

        # Level 3: PTE
        pte_addr = pmd_entry & self._PTE_ADDR_MASK
        pte_entry = self._read_phys_u64(pte_addr + pte_idx * 8)
        if not (pte_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PTE entry"
            )

        page_phys = pte_entry & self._PTE_ADDR_MASK

        # Cache the translation
        self._translation_cache[page_vaddr] = (page_phys, self._PAGE_SIZE, pte_entry)

        return page_phys + (vaddr & page_mask), self._PAGE_SIZE, pte_entry

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
        """Returns a sorted iterable of (offset, sublength, mapped_offset, mapped_length, layer)
        mappings.

        This allows translation layers to provide maps of contiguous regions in one layer.
        """
        if length == 0:
            try:
                mapped_offset, _, layer_name = self._translate(offset)
                if not self._context.layers[layer_name].is_valid(mapped_offset):
                    raise exceptions.InvalidAddressException(
                        layer_name=layer_name, invalid_address=mapped_offset
                    )
            except exceptions.InvalidAddressException:
                if not ignore_errors:
                    raise
                return None
            yield offset, length, mapped_offset, length, layer_name
            return None

        while length > 0:
            try:
                chunk_offset, page_size, layer_name = self._translate(offset)
                chunk_size = min(page_size - (offset % page_size), length)
                if not self._context.layers[layer_name].is_valid(
                    chunk_offset, chunk_size
                ):
                    raise exceptions.InvalidAddressException(
                        layer_name=layer_name, invalid_address=chunk_offset
                    )
            except exceptions.InvalidAddressException:
                if not ignore_errors:
                    raise
                # Skip to next page boundary
                skip_size = self._PAGE_SIZE - (offset % self._PAGE_SIZE)
                length -= skip_size
                offset += skip_size
            else:
                yield offset, chunk_size, chunk_offset, chunk_size, layer_name
                length -= chunk_size
                offset += chunk_size

    def canonicalize(self, addr: int) -> int:
        """Canonicalizes an address by sign-extending from the VA width.

        AArch64 kernel addresses have upper bits set (TTBR1 space).
        For 48-bit VA, addresses >= 0xFFFF000000000000 are kernel space.
        For 47-bit VA (16KB pages), addresses >= 0xFFFF800000000000 are kernel space.
        For 39-bit VA (4KB pages), addresses >= 0xFFFFFF8000000000 are kernel space.
        """
        if self._page_table_levels == 3:
            if self._PAGE_SHIFT == 14:
                # 47-bit VA (16KB pages): sign extend from bit 46
                if addr & (1 << 46):
                    return addr | 0xFFFF800000000000
            else:
                # 39-bit VA (4KB pages): sign extend from bit 38
                if addr & (1 << 38):
                    return addr | 0xFFFFFF8000000000
        else:
            # 48-bit VA: sign extend from bit 47
            if addr & (1 << 47):
                return addr | 0xFFFF000000000000
        return addr


class LinuxAArch64(AArch64):
    """Linux-specific AArch64 layer."""
