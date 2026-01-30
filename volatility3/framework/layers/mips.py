# This file is Copyright 2024 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

import struct
from typing import List, Tuple

from volatility3.framework import interfaces, exceptions
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import linear


class MIPS64(linear.LinearlyMappedLayer):
    """MIPS64 translation layer.

    Translates virtual addresses to physical using the page global directory.
    Supports MIPS64 Linux with 3-level page tables (PGD -> PMD -> PTE).

    MIPS64 memory segments:
    - XKPHYS (0x8000000000000000 - 0xbfffffffffffffff): Direct physical mapping
    - CKSEG0 (0xffffffff80000000 - 0xffffffff9fffffff): Cached unmapped (512MB)
    - CKSEG1 (0xffffffffa0000000 - 0xffffffffbfffffff): Uncached unmapped (512MB)
    - KSEG2/KSEG3 (0xffffffffc0000000 - 0xffffffffffffffff): Mapped kernel space
    """

    _direct_metadata = {
        "architecture": "MIPS64",
        "mapped": True,
    }

    # MIPS64 is big-endian and uses 64-bit entries
    _entry_format = ">Q"  # Big-endian 64-bit
    _bits_per_register = 64

    # Page table constants for 4KB pages
    _PAGE_SHIFT = 12
    _PAGE_SIZE = 1 << 12  # 4096
    _PAGE_MASK = ~(_PAGE_SIZE - 1)

    # MIPS64 PTE flags
    _PTE_VALID = 1 << 1  # Valid bit (bit 1 in MIPS)
    _PTE_DIRTY = 1 << 2  # Dirty/Modified bit
    _PTE_GLOBAL = 1 << 0  # Global bit

    # Page table index bits for 3-level (PGD -> PMD -> PTE)
    # With 4KB pages and 8-byte entries:
    # - PTE: 512 entries (9 bits) covering 2MB
    # - PMD: 512 entries (9 bits) covering 1GB
    # - PGD: varies based on VA width
    _PTRS_PER_PTE = 512
    _PTRS_PER_PMD = 512
    _PTRS_PER_PGD = 512

    _PTE_SHIFT = 12
    _PMD_SHIFT = 21  # 12 + 9
    _PGD_SHIFT = 30  # 12 + 9 + 9

    # Physical address mask in PTE (bits [63:12] typically, but varies)
    # For Cavium Octeon, physical addresses are in bits [63:12]
    _PTE_PFN_MASK = 0xFFFFFFFFFFFFF000

    # MIPS64 memory segment bases
    _CKSEG0_BASE = 0xFFFFFFFF80000000
    _CKSEG1_BASE = 0xFFFFFFFFA0000000
    _KSEG2_BASE = 0xFFFFFFFFC0000000
    _XKPHYS_BASE = 0x8000000000000000

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.TranslationLayerRequirement(
                name="memory_layer", optional=False
            ),
            requirements.IntRequirement(name="page_map_offset", optional=False),
            requirements.IntRequirement(name="kernel_virtual_offset", optional=True),
            requirements.StringRequirement(name="kernel_banner", optional=True),
        ]

    def __init__(self, context, config_path, name, metadata=None):
        super().__init__(
            context=context, config_path=config_path, name=name, metadata=metadata
        )
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
        """Read a 64-bit big-endian value from physical memory."""
        try:
            data = self.context.layers[self._base_layer].read(phys_addr, 8)
            return struct.unpack(">Q", data)[0]
        except exceptions.InvalidAddressException:
            return 0

    def _is_direct_mapped(self, vaddr: int) -> bool:
        """Check if address is in a directly mapped segment (CKSEG0/CKSEG1/XKPHYS)."""
        # CKSEG0: 0xffffffff80000000 - 0xffffffff9fffffff
        if vaddr >= self._CKSEG0_BASE and vaddr < self._CKSEG1_BASE:
            return True
        # CKSEG1: 0xffffffffa0000000 - 0xffffffffbfffffff
        if vaddr >= self._CKSEG1_BASE and vaddr < self._KSEG2_BASE:
            return True
        # XKPHYS: 0x8000000000000000 - 0xbfffffffffffffff
        if vaddr >= self._XKPHYS_BASE and vaddr < 0xC000000000000000:
            return True
        return False

    def _direct_map_translate(self, vaddr: int) -> int:
        """Translate directly mapped addresses."""
        # CKSEG0/CKSEG1: virtual 0xffffffff8xxxxxxx -> physical 0x0xxxxxxx
        if vaddr >= self._CKSEG0_BASE and vaddr < self._KSEG2_BASE:
            return vaddr & 0x1FFFFFFF  # Lower 29 bits (512MB)
        # XKPHYS: extract physical address from bits [58:0] with cache coherency in [61:59]
        if vaddr >= self._XKPHYS_BASE and vaddr < 0xC000000000000000:
            return vaddr & 0x07FFFFFFFFFFFFFF  # Lower 59 bits
        raise exceptions.InvalidAddressException(
            self.name, vaddr, "Not a directly mapped address"
        )

    def _translate_entry(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate virtual address and return (physical_addr, page_size, pte_entry).

        MIPS64 Linux uses 3-level page tables: PGD -> PMD -> PTE
        For kernel addresses, most are in directly mapped segments (CKSEG0/CKSEG1/XKPHYS).
        """
        # Check for directly mapped addresses first (most common case for kernel)
        if self._is_direct_mapped(vaddr):
            phys = self._direct_map_translate(vaddr)
            return phys, self._PAGE_SIZE, 0

        # For non-direct-mapped addresses, we would need page table translation
        # Currently only supporting direct-mapped kernel addresses
        # User-space addresses and KSEG2/KSEG3 mapped addresses are not yet supported
        raise exceptions.InvalidAddressException(
            self.name,
            vaddr,
            "Address not in direct-mapped region (CKSEG0/CKSEG1/XKPHYS)",
        )

    def _translate_3level(self, vaddr: int) -> Tuple[int, int, int]:
        """Translate using 3-level page table (PGD -> PMD -> PTE)."""
        page_mask = self._PAGE_SIZE - 1
        page_vaddr = vaddr & ~page_mask

        # Extract indices
        pgd_idx = (vaddr >> self._PGD_SHIFT) & (self._PTRS_PER_PGD - 1)
        pmd_idx = (vaddr >> self._PMD_SHIFT) & (self._PTRS_PER_PMD - 1)
        pte_idx = (vaddr >> self._PTE_SHIFT) & (self._PTRS_PER_PTE - 1)

        # Level 0: PGD
        pgd_entry_addr = self._pgd_addr + pgd_idx * 8
        pgd_entry = self._read_phys_u64(pgd_entry_addr)
        if pgd_entry == 0:
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PGD entry (null)"
            )

        # PGD entry contains physical address of PMD table
        # On MIPS, the PGD entry is typically a direct pointer
        pmd_addr = pgd_entry & self._PTE_PFN_MASK

        # Level 1: PMD
        pmd_entry = self._read_phys_u64(pmd_addr + pmd_idx * 8)
        if pmd_entry == 0:
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PMD entry (null)"
            )

        # PMD entry contains physical address of PTE table
        pte_addr = pmd_entry & self._PTE_PFN_MASK

        # Level 2: PTE
        pte_entry = self._read_phys_u64(pte_addr + pte_idx * 8)
        if not (pte_entry & self._PTE_VALID):
            raise exceptions.InvalidAddressException(
                self.name, vaddr, "Invalid PTE entry (not valid)"
            )

        # Extract physical page frame number
        # MIPS PTE format: PFN is in upper bits, flags in lower bits
        # The exact format depends on the MIPS variant
        # For standard MIPS64: PFN starts at bit 6 or higher
        page_phys = (pte_entry >> 6) << self._PAGE_SHIFT

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
        """Canonicalizes an address by sign-extending.

        MIPS64 kernel addresses are typically in the upper half of the address space.
        """
        # MIPS64 addresses are already canonical in most cases
        # Sign-extend from bit 63 if needed
        if addr & (1 << 63):
            return addr | 0xFFFFFFFF00000000
        return addr


class LinuxMIPS64(MIPS64):
    """Linux-specific MIPS64 layer."""

