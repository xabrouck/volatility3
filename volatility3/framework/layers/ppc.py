# This file is Copyright 2024 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

from typing import List, Optional, Tuple

from volatility3.framework import interfaces, exceptions
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import linear


class PPC32(linear.LinearlyMappedLayer):
    """PowerPC 32-bit translation layer.

    Translates virtual addresses to physical using linear mapping.
    PPC32 Linux (especially Book-E/e500) uses a simple linear mapping for kernel space:
    - PAGE_OFFSET (typically 0xc0000000) is the start of kernel virtual address space
    - Physical address = Virtual address - PAGE_OFFSET + physical_offset

    The physical_offset accounts for relocatable kernels where the kernel may be
    loaded at a non-zero physical address.
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
            requirements.StringRequirement(name="kernel_banner", optional=True),
        ]

    def __init__(self, context, config_path, name, metadata=None):
        super().__init__(
            context=context, config_path=config_path, name=name, metadata=metadata
        )
        self._base_layer = self.config["memory_layer"]
        self._page_offset = self.config.get("kernel_virtual_offset", self._PAGE_OFFSET)
        self._phys_offset = self.config.get("kernel_physical_offset", 0)

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

    def is_valid(self, offset: int, length: int = 1) -> bool:
        """Check if the virtual address range is valid."""
        try:
            return self._translate(offset) is not None
        except exceptions.InvalidAddressException:
            return False

    def _translate(self, vaddr: int) -> Optional[int]:
        """Translate virtual address to physical address.

        PPC32 Linux uses linear mapping for kernel addresses:
        phys = virt - PAGE_OFFSET + phys_offset
        """
        # Ensure address is in kernel space
        if vaddr < self._page_offset:
            # User space address - would need page table translation
            # For now, we don't support user space
            return None

        # Linear mapping for kernel space
        phys = (vaddr - self._page_offset) + self._phys_offset

        # Verify the physical address is valid in the underlying layer
        if phys < 0:
            return None

        return phys

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
        base_layer = self._context.layers[self._base_layer]
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
