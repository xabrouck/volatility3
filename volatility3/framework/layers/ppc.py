# This file is Copyright 2024 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

import logging
import struct
from typing import Dict, List, Optional, Tuple

from volatility3.framework import interfaces, exceptions
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import linear

vollog = logging.getLogger(__name__)


class PPC32(linear.LinearlyMappedLayer):
    """PowerPC 32-bit translation layer.

    Translates virtual addresses to physical using:
    1. Linear mapping for lowmem (PAGE_OFFSET to VMALLOC_START)
    2. Page-based translation for vmalloc region (using vmap_area_list)

    PPC32 Linux (especially Book-E/e500) uses software TLB management,
    so vmalloc addresses require looking up the physical pages through
    the vmap_area_list and mem_map structures.
    """

    _direct_metadata = {
        "architecture": "PPC32",
        "mapped": True,
    }

    # PPC32 uses big-endian 32-bit entries
    _entry_format = ">I"
    _bits_per_register = 32

    # Page size (4KB typical)
    _PAGE_SHIFT = 12
    _PAGE_SIZE = 1 << 12  # 4096

    # Default PAGE_OFFSET for PPC32 Linux
    _PAGE_OFFSET = 0xC0000000

    # Typical vmalloc start for PPC32 (can vary based on kernel config)
    _VMALLOC_START = 0xF0000000

    # PPC32 nohash (Book-E) page table constants
    # 2-level page table: PGD (1024 entries) -> PTE (1024 entries, 64-bit each)
    _PGD_SHIFT = 22  # 10 bits for PGD index (1024 entries)
    _PTE_SHIFT = 12  # 10 bits for PTE index (1024 entries)
    _PGD_SIZE = 4    # 32-bit PGD entries
    _PTE_SIZE = 8    # 64-bit PTE entries

    # PTE flags for PPC32 nohash (in lower 32 bits of 64-bit PTE)
    _PAGE_PRESENT = 0x1      # Bit 0: Present/Valid
    _PAGE_DIRTY = 0x100      # Bit 8: Dirty (page has been written)
    _PTE_RPN_SHIFT = 12      # RPN starts at bit 12 in 64-bit PTE

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.TranslationLayerRequirement(
                name="memory_layer", optional=False
            ),
            requirements.IntRequirement(
                name="kernel_virtual_offset",
                description="Virtual address offset (PAGE_OFFSET)",
                optional=True,
                default=0xC0000000,
            ),
            requirements.IntRequirement(
                name="kernel_physical_offset",
                description="Physical address where kernel is loaded",
                optional=True,
                default=0,
            ),
            requirements.IntRequirement(
                name="vmalloc_start",
                description="Start of vmalloc region",
                optional=True,
                default=0xF0000000,
            ),
            requirements.IntRequirement(
                name="vmap_area_list",
                description="Virtual address of vmap_area_list symbol",
                optional=True,
                default=0,
            ),
            requirements.IntRequirement(
                name="mem_map",
                description="Virtual address of mem_map symbol",
                optional=True,
                default=0,
            ),
            requirements.IntRequirement(
                name="page_struct_size",
                description="Size of struct page",
                optional=True,
                default=36,
            ),
            requirements.IntRequirement(
                name="page_map_offset",
                description="DTB/PGD physical address for process layer",
                optional=True,
                default=0,
            ),
            requirements.StringRequirement(name="kernel_banner", optional=True),
        ]

    def __init__(self, context, config_path, name, metadata=None):
        super().__init__(
            context=context, config_path=config_path, name=name, metadata=metadata
        )
        self._base_layer = self.config["memory_layer"]
        # PAGE_OFFSET is always 0xC0000000 for PPC32 Linux memory layout
        # (kernel_virtual_offset config is for symbol offset adjustment, not memory layout)
        self._page_offset = self._PAGE_OFFSET
        self._phys_offset = self.config.get("kernel_physical_offset", 0)
        self._vmalloc_start = self.config.get("vmalloc_start", self._VMALLOC_START)
        self._vmap_area_list = self.config.get("vmap_area_list", 0)
        self._mem_map_ptr = self.config.get("mem_map", 0)
        self._page_struct_size = self.config.get("page_struct_size", 36)
        self._page_map_offset = self.config.get("page_map_offset", 0)

        if self._page_map_offset:
            vollog.debug(f"PPC32 layer {name}: page_map_offset={hex(self._page_map_offset)}")

        # Cache for vmalloc translations: {vaddr_page: phys_page}
        self._vmalloc_cache: Dict[int, int] = {}
        self._vmalloc_cache_built = False
        self._mem_map_value: Optional[int] = None

    @property
    def dependencies(self) -> List[str]:
        return [self._base_layer]

    @property
    def minimum_address(self) -> int:
        return 0

    @property
    def maximum_address(self) -> int:
        return (2**32) - 1

    @property
    def page_shift(self) -> int:
        return self._PAGE_SHIFT

    @property
    def page_size(self) -> int:
        return self._PAGE_SIZE

    @property
    def bits_per_register(self) -> int:
        return self._bits_per_register

    def is_valid(self, offset: int, length: int = 1) -> bool:
        """Check if the virtual address range is valid."""
        try:
            return self._translate(offset) is not None
        except exceptions.InvalidAddressException:
            return False

    def is_dirty(self, offset: int) -> bool:
        """Returns whether the page at offset is marked dirty.

        For user-space addresses with page tables, checks the PTE dirty bit.
        For kernel linear mapping, dirty tracking isn't available.
        """
        # User-space address - check PTE dirty bit if page tables available
        if offset < self._page_offset and self._page_map_offset != 0:
            pte = self._get_pte(offset)
            if pte is not None:
                return (pte & self._PAGE_DIRTY) != 0
        # Kernel space or no page tables - no dirty tracking
        return False

    def _get_pte(self, vaddr: int) -> Optional[int]:
        """Get the PTE entry for a user-space virtual address.

        Returns the 64-bit PTE value or None if not mapped.
        """
        return self._read_pte_entry(vaddr)

    def _read_phys_u32(self, phys_addr: int) -> Optional[int]:
        """Read a big-endian 32-bit value from physical memory."""
        try:
            base_layer = self._context.layers[self._base_layer]
            data = base_layer.read(phys_addr, 4)
            return struct.unpack(">I", data)[0]
        except Exception:
            return None

    def _linear_translate(self, vaddr: int) -> Optional[int]:
        """Translate using linear mapping (for lowmem addresses)."""
        if vaddr < self._page_offset:
            return None
        phys = (vaddr - self._page_offset) + self._phys_offset
        if phys < 0:
            return None
        return phys

    def _get_mem_map(self) -> Optional[int]:
        """Get the mem_map array base address."""
        if self._mem_map_value is not None:
            return self._mem_map_value

        if not self._mem_map_ptr:
            return None

        # mem_map is a pointer, read its value
        mem_map_phys = self._linear_translate(self._mem_map_ptr)
        if mem_map_phys is None:
            return None

        self._mem_map_value = self._read_phys_u32(mem_map_phys)
        return self._mem_map_value

    def _page_to_pfn(self, page_ptr: int) -> Optional[int]:
        """Convert a struct page pointer to a page frame number."""
        mem_map = self._get_mem_map()
        if mem_map is None or self._page_struct_size == 0:
            return None

        if page_ptr < mem_map:
            return None

        pfn = (page_ptr - mem_map) // self._page_struct_size
        return pfn

    def _build_vmalloc_cache_for_range(self, va_start: int, va_end: int, vm_ptr: int) -> None:
        """Build vmalloc cache entries for a specific vmap_area range."""
        # vm_struct offsets (PPC32 big-endian):
        # addr: offset 4
        # size: offset 8
        # pages: offset 16
        # nr_pages: offset 20
        # phys_addr: offset 24

        vm_phys = self._linear_translate(vm_ptr)
        if vm_phys is None:
            return

        try:
            base_layer = self._context.layers[self._base_layer]
            vm_data = base_layer.read(vm_phys, 28)

            vm_addr = struct.unpack(">I", vm_data[4:8])[0]
            vm_size = struct.unpack(">I", vm_data[8:12])[0]
            vm_pages_ptr = struct.unpack(">I", vm_data[16:20])[0]
            vm_nr_pages = struct.unpack(">I", vm_data[20:24])[0]
            vm_phys_addr = struct.unpack(">I", vm_data[24:28])[0]

            # If phys_addr is set, use direct mapping
            if vm_phys_addr != 0:
                for page_idx in range(vm_nr_pages):
                    vpage = (vm_addr + page_idx * self._PAGE_SIZE) & ~(self._PAGE_SIZE - 1)
                    ppage = vm_phys_addr + page_idx * self._PAGE_SIZE
                    self._vmalloc_cache[vpage] = ppage
                return

            # Otherwise, use pages array
            if vm_pages_ptr == 0 or vm_nr_pages == 0:
                return

            pages_phys = self._linear_translate(vm_pages_ptr)
            if pages_phys is None:
                return

            # Read page pointers array
            pages_data = base_layer.read(pages_phys, vm_nr_pages * 4)

            for page_idx in range(vm_nr_pages):
                page_ptr = struct.unpack(">I", pages_data[page_idx * 4:(page_idx + 1) * 4])[0]
                if page_ptr == 0:
                    continue

                pfn = self._page_to_pfn(page_ptr)
                if pfn is None:
                    continue

                vpage = (vm_addr + page_idx * self._PAGE_SIZE) & ~(self._PAGE_SIZE - 1)
                ppage = pfn * self._PAGE_SIZE
                self._vmalloc_cache[vpage] = ppage

        except Exception as e:
            vollog.debug(f"Error building vmalloc cache for vm_struct at {hex(vm_ptr)}: {e}")

    def _build_vmalloc_cache(self) -> None:
        """Build the vmalloc translation cache from vmap_area_list."""
        if self._vmalloc_cache_built:
            return

        self._vmalloc_cache_built = True

        if not self._vmap_area_list:
            vollog.debug("No vmap_area_list configured, vmalloc translation disabled")
            return

        # vmap_area_list is a list_head
        # vmap_area structure offsets:
        # va_start: 0, va_end: 4, flags: 8, rb_node: 12, list: 24, vm: 36

        list_head_phys = self._linear_translate(self._vmap_area_list)
        if list_head_phys is None:
            vollog.debug("Cannot translate vmap_area_list address")
            return

        try:
            base_layer = self._context.layers[self._base_layer]
            list_data = base_layer.read(list_head_phys, 8)
            list_next = struct.unpack(">I", list_data[0:4])[0]

            current = list_next
            count = 0
            max_entries = 10000  # Safety limit

            while current != self._vmap_area_list and count < max_entries:
                count += 1

                # list entry is at offset 24 in vmap_area
                vmap_area_vaddr = current - 24
                vmap_area_phys = self._linear_translate(vmap_area_vaddr)

                if vmap_area_phys is None:
                    break

                # Read vmap_area fields
                va_data = base_layer.read(vmap_area_phys, 40)
                va_start = struct.unpack(">I", va_data[0:4])[0]
                va_end = struct.unpack(">I", va_data[4:8])[0]
                list_next_new = struct.unpack(">I", va_data[24:28])[0]
                vm_ptr = struct.unpack(">I", va_data[36:40])[0]

                # Build cache for this vmap_area if it has a vm_struct
                if vm_ptr != 0 and va_start >= self._vmalloc_start:
                    self._build_vmalloc_cache_for_range(va_start, va_end, vm_ptr)

                current = list_next_new

            vollog.debug(f"Built vmalloc cache with {len(self._vmalloc_cache)} page mappings from {count} vmap_areas")

        except Exception as e:
            vollog.debug(f"Error building vmalloc cache: {e}")

    def _vmalloc_translate(self, vaddr: int) -> Optional[int]:
        """Translate a vmalloc address using the cache."""
        if not self._vmalloc_cache_built:
            self._build_vmalloc_cache()

        vpage = vaddr & ~(self._PAGE_SIZE - 1)
        page_offset = vaddr & (self._PAGE_SIZE - 1)

        ppage = self._vmalloc_cache.get(vpage)
        if ppage is None:
            return None

        return ppage + page_offset

    def _read_pte_entry(self, vaddr: int) -> Optional[int]:
        """Read the PTE entry for a virtual address.

        Returns the raw 64-bit PTE value or None if not mapped.
        Used by both _pagetable_translate and _get_pte (for is_dirty).
        """
        if self._page_map_offset == 0:
            return None

        try:
            base_layer = self._context.layers[self._base_layer]

            # Calculate PGD index and read entry
            pgd_index = (vaddr >> self._PGD_SHIFT) & 0x3FF
            pgd_entry_addr = self._page_map_offset + (pgd_index * self._PGD_SIZE)
            pgd_data = base_layer.read(pgd_entry_addr, 4)
            pgd_entry = struct.unpack(">I", pgd_data)[0]

            if pgd_entry == 0:
                return None

            # PGD entry contains VIRTUAL address of PTE table (kernel pointer)
            # Convert to physical using linear mapping
            pte_table_virt = pgd_entry & ~(self._PAGE_SIZE - 1)
            if pte_table_virt >= self._PAGE_OFFSET:
                pte_table_phys = pte_table_virt - self._PAGE_OFFSET
            else:
                pte_table_phys = pte_table_virt

            # Calculate PTE index and read entry
            pte_index = (vaddr >> self._PTE_SHIFT) & 0x3FF
            pte_entry_addr = pte_table_phys + (pte_index * self._PTE_SIZE)
            pte_data = base_layer.read(pte_entry_addr, 8)
            return struct.unpack(">Q", pte_data)[0]

        except Exception:
            return None

    def _pte_to_phys(self, pte_entry: int) -> int:
        """Extract physical page address from a PTE entry.

        PPC32 e500/Book-E with CONFIG_PTE_64BIT format:
        RPN spans both words: upper_word has high bits, upper byte of lower_word has low bits.
        """
        upper_word = (pte_entry >> 32) & 0xFFFFFFFF
        lower_word = pte_entry & 0xFFFFFFFF
        rpn = (upper_word << 8) | ((lower_word >> 24) & 0xFF)
        return rpn << 12

    def _pagetable_translate(self, vaddr: int) -> Optional[int]:
        """Translate user-space address using page table walking.

        PPC32 nohash (Book-E) uses a 2-level page table:
        - PGD: 1024 entries, 4 bytes each (32-bit pointers to PTE tables)
        - PTE: 1024 entries, 8 bytes each (64-bit PTEs with RPN and flags)
        """
        pte_entry = self._read_pte_entry(vaddr)
        if pte_entry is None:
            return None

        # Check if PTE is valid (present)
        if (pte_entry & self._PAGE_PRESENT) == 0:
            return None

        phys_page = self._pte_to_phys(pte_entry)
        page_offset = vaddr & (self._PAGE_SIZE - 1)
        return phys_page + page_offset

    def _translate(self, vaddr: int) -> Optional[int]:
        """Translate virtual address to physical address.

        PPC32 Linux uses:
        - Page table walking for user space (if page_map_offset is set)
        - Linear mapping for lowmem (PAGE_OFFSET to VMALLOC_START)
        - Page-based translation for vmalloc region (VMALLOC_START and above)
        """
        # User space address - use page table translation if available
        if vaddr < self._page_offset:
            return self._pagetable_translate(vaddr)

        # Check if address is in vmalloc region
        if vaddr >= self._vmalloc_start:
            return self._vmalloc_translate(vaddr)

        # Linear mapping for lowmem
        return self._linear_translate(vaddr)

    def mapping(
        self, offset: int, length: int, ignore_errors: bool = False
    ) -> List[Tuple[int, int, int, int, str]]:
        """Map virtual addresses to physical addresses.

        Returns: List of (offset, sublength, mapped_offset, mapped_length, layer)
        """
        result = []

        phys = self._translate(offset)
        if phys is None:
            if not ignore_errors:
                raise exceptions.InvalidAddressException(
                    layer_name=self.name,
                    invalid_address=offset,
                )
            return result

        # Check if the physical address is valid in the base layer
        # Handle length=0 case (used by translate() to check if address is valid)
        base_layer = self._context.layers[self._base_layer]
        if length == 0:
            if not base_layer.is_valid(phys):
                if not ignore_errors:
                    raise exceptions.InvalidAddressException(
                        layer_name=self.name,
                        invalid_address=offset,
                    )
                return result
        else:
            if not base_layer.is_valid(phys, length):
                if not ignore_errors:
                    raise exceptions.InvalidAddressException(
                        layer_name=self.name,
                        invalid_address=offset,
                    )
                return result

        # Format: (offset, sublength, mapped_offset, mapped_length, layer)
        result.append((offset, length, phys, length, self._base_layer))
        return result

    def canonicalize(self, addr: int) -> int:
        """Canonicalizes an address.

        PPC32 addresses are 32-bit, no canonicalization needed.
        """
        return addr & 0xFFFFFFFF
