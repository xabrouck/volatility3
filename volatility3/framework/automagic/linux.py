# This file is Copyright 2019 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

import logging
from typing import Optional, Tuple

from volatility3.framework import constants, interfaces
from volatility3.framework.automagic import symbol_cache, symbol_finder
from volatility3.framework.configuration import requirements
from volatility3.framework.layers import arm, intel, mips, ppc, scanners
from volatility3.framework.symbols import linux

vollog = logging.getLogger(__name__)


class LinuxIntelStacker(interfaces.automagic.StackerLayerInterface):
    stack_order = 35
    exclusion_list = ["mac", "windows"]

    @classmethod
    def stack(
        cls,
        context: interfaces.context.ContextInterface,
        layer_name: str,
        progress_callback: constants.ProgressCallback = None,
    ) -> Optional[interfaces.layers.DataLayerInterface]:
        """Attempts to identify linux within this layer."""
        # Bail out by default unless we can stack properly
        layer = context.layers[layer_name]
        join = interfaces.configuration.path_join

        # Never stack on top of a translation layer (Intel, AArch64, MIPS64, or PPC32)
        if isinstance(layer, (intel.Intel, arm.AArch64, mips.MIPS64, ppc.PPC32)):
            return None

        linux_banners = symbol_cache.load_cache_manager().get_identifier_dictionary(
            operating_system="linux"
        )
        # If we have no banners, don't bother scanning
        if not linux_banners:
            vollog.info(
                "No Linux banners found - if this is a linux plugin, please check your symbol files location"
            )
            return None

        mss = scanners.MultiStringScanner([x for x in linux_banners if x is not None])
        for _, banner in layer.scan(
            context=context, scanner=mss, progress_callback=progress_callback
        ):
            dtb = None
            vollog.debug(f"Identified banner: {repr(banner)}")

            isf_path = linux_banners.get(banner, None)
            if isf_path:
                table_name = context.symbol_space.free_table_name("LintelStacker")
                table = linux.LinuxKernelIntermedSymbols(
                    context,
                    "temporary." + table_name,
                    name=table_name,
                    isf_url=isf_path,
                )
                context.symbol_space.append(table)

                kaslr_shift, aslr_shift = cls.find_aslr(
                    context,
                    table_name,
                    layer_name,
                    progress_callback=progress_callback,
                )

                # Skip non-Intel kernels by checking for x86-specific symbols
                # idt_table (Interrupt Descriptor Table) is x86-only; ARM uses GIC/exception vectors
                if "idt_table" not in table.symbols:
                    vollog.debug("Skipping Intel stacker: idt_table symbol not found")
                    continue

                if "init_top_pgt" in table.symbols:
                    layer_class = intel.LinuxIntel32e
                    dtb_symbol_name = "init_top_pgt"
                elif "init_level4_pgt" in table.symbols:
                    layer_class = intel.LinuxIntel32e
                    dtb_symbol_name = "init_level4_pgt"
                elif "pkmap_count" in table.symbols and table.get_symbol(
                    "pkmap_count"
                ).type.count in (512, 2048):
                    layer_class = intel.LinuxIntelPAE
                    dtb_symbol_name = "swapper_pg_dir"
                else:
                    layer_class = intel.LinuxIntel
                    dtb_symbol_name = "swapper_pg_dir"

                dtb = cls.virtual_to_physical_address(
                    table.get_symbol(dtb_symbol_name).address + kaslr_shift
                )

                # Build the new layer
                new_layer_name = context.layers.free_layer_name("IntelLayer")
                config_path = join("IntelHelper", new_layer_name)
                context.config[join(config_path, "memory_layer")] = layer_name
                context.config[join(config_path, "page_map_offset")] = dtb
                context.config[
                    join(config_path, LinuxSymbolFinder.banner_config_key)
                ] = str(banner, "latin-1")

                layer = layer_class(
                    context,
                    config_path=config_path,
                    name=new_layer_name,
                    metadata={"os": "Linux"},
                )
                layer.config["kernel_virtual_offset"] = aslr_shift

            if layer and dtb:
                vollog.debug(f"DTB was found at: 0x{dtb:0x}")
                return layer
        vollog.debug("No suitable linux banner could be matched")
        return None

    @classmethod
    def find_aslr(
        cls,
        context: interfaces.context.ContextInterface,
        symbol_table: str,
        layer_name: str,
        progress_callback: constants.ProgressCallback = None,
    ) -> Tuple[int, int]:
        """Determines the offset of the actual DTB in physical space and its
        symbol offset.

        Args:
            context: The context to retrieve required elements (layers, symbol tables) from
            symbol_table: The name of the kernel module on which to operate
            layer_name: The layer within the context in which the module exists
            progress_callback: A function that takes a percentage (and an optional description) that will be called periodically

        Returns:
            kaslr_shirt and aslr_shift
        """
        init_task_symbol = symbol_table + constants.BANG + "init_task"
        init_task_json_address = context.symbol_space.get_symbol(
            init_task_symbol
        ).address
        swapper_signature = rb"swapper(\/0|\x00\x00)\x00\x00\x00\x00\x00\x00"
        module = context.module(symbol_table, layer_name, 0)
        address_mask = context.symbol_space[symbol_table].config.get(
            "symbol_mask", None
        )

        task_symbol = module.get_type("task_struct")
        comm_child_offset = task_symbol.relative_child_offset("comm")

        for offset in context.layers[layer_name].scan(
            scanner=scanners.RegExScanner(swapper_signature),
            context=context,
            progress_callback=progress_callback,
        ):
            init_task_address = offset - comm_child_offset
            init_task = module.object(
                object_type="task_struct", offset=init_task_address, absolute=True
            )
            if init_task.pid != 0:
                continue
            elif (
                init_task.has_member("state")
                and init_task.state.cast("unsigned int") != 0
            ):
                continue
            elif init_task.active_mm.cast("long unsigned int") == module.get_symbol(
                "init_mm"
            ).address and init_task.tasks.next.cast(
                "long unsigned int"
            ) == init_task.tasks.prev.cast(
                "long unsigned int"
            ):
                # The idle task steals `mm` from previously running task, i.e.,
                # `init_mm` is only used as long as no CPU has ever been idle.
                # This catches cases where we found a fragment of the
                # unrelocated ELF file instead of the running kernel.
                continue

            # This we get for free
            aslr_shift = (
                int.from_bytes(
                    init_task.files.cast("bytes", length=init_task.files.vol.size),
                    byteorder=init_task.files.vol.data_format.byteorder,
                )
                - module.get_symbol("init_files").address
            )
            kaslr_shift = init_task_address - cls.virtual_to_physical_address(
                init_task_json_address
            )
            if address_mask:
                aslr_shift = aslr_shift & address_mask

            if aslr_shift & 0xFFF != 0 or kaslr_shift & 0xFFF != 0:
                continue
            vollog.debug(
                f"Linux ASLR shift values determined: physical {kaslr_shift:0x} virtual {aslr_shift:0x}"
            )
            return kaslr_shift, aslr_shift

        # We don't throw an exception, because we may legitimately not have an ASLR shift, but we report it
        vollog.debug("Scanners could not determine any ASLR shifts, using 0 for both")
        return 0, 0

    @staticmethod
    def virtual_to_physical_address(addr: int) -> int:
        """Converts a virtual linux address to a physical one (does not account
        of ASLR)"""
        if addr > 0xFFFFFFFF80000000:
            return addr - 0xFFFFFFFF80000000
        return addr - 0xC0000000


class LinuxAArch64Stacker(interfaces.automagic.StackerLayerInterface):
    stack_order = 35
    exclusion_list = ["mac", "windows"]

    @classmethod
    def stack(
        cls,
        context: interfaces.context.ContextInterface,
        layer_name: str,
        progress_callback: constants.ProgressCallback = None,
    ) -> Optional[interfaces.layers.DataLayerInterface]:
        """Attempts to identify linux AArch64 within this layer."""
        layer = context.layers[layer_name]
        join = interfaces.configuration.path_join

        # Never stack on top of an existing translation layer
        if isinstance(layer, (intel.Intel, arm.AArch64, mips.MIPS64, ppc.PPC32)):
            return None

        linux_banners = symbol_cache.load_cache_manager().get_identifier_dictionary(
            operating_system="linux"
        )
        if not linux_banners:
            vollog.info(
                "No Linux banners found - if this is a linux plugin, please check your symbol files location"
            )
            return None

        mss = scanners.MultiStringScanner([x for x in linux_banners if x is not None])
        for _, banner in layer.scan(
            context=context, scanner=mss, progress_callback=progress_callback
        ):
            vollog.debug(f"Identified banner: {repr(banner)}")

            isf_path = linux_banners.get(banner, None)
            if not isf_path:
                continue

            table_name = context.symbol_space.free_table_name("LinuxAArch64Stacker")
            table = linux.LinuxKernelIntermedSymbols(
                context,
                "temporary." + table_name,
                name=table_name,
                isf_url=isf_path,
            )
            context.symbol_space.append(table)

            # Check if this is an AArch64 kernel
            if "swapper_pg_dir" not in table.symbols:
                continue

            # Skip 32-bit kernels (PPC32, ARM32, etc.) - AArch64 uses 64-bit pointers
            ptr_type = context.symbol_space.get_type(table_name + constants.BANG + "pointer")
            if ptr_type.size != 8:
                vollog.debug(
                    f"Skipping AArch64 stacker: pointer size is {ptr_type.size}, not 8"
                )
                continue

            # Skip MIPS64 kernels - check for MIPS-specific TLB symbol
            if "r4k_tlb_init_pm" in table.symbols:
                vollog.debug(
                    "Skipping AArch64 stacker: MIPS64 kernel detected (r4k_tlb_init_pm)"
                )
                continue

            # For AArch64, swapper_pg_dir is the page global directory
            swapper_pg_dir_symbol = table.get_symbol("swapper_pg_dir")

            # Determine page table levels from symbol table
            # If __pud_alloc exists, PUD is a real level (4-level page tables)
            # If __pud_alloc is missing, PUD is folded into PGD (3-level page tables)
            if "__pud_alloc" in table.symbols:
                page_table_levels = 4
            else:
                page_table_levels = 3

            # Detect page size from kernel configuration BEFORE calculating ASLR
            # 16KB pages use different VA bits and page table structure
            # Key insight: For 16KB pages with 47-bit VA, we have 3-level page tables
            # but the linear map starts at 0xFFFFC00000000000 (not 0xFFFF800000000000)
            # For 4KB pages with 39-bit VA, we also have 3-level but linear map at 0xFFFFFF8000000000
            # For 4KB pages with 48-bit VA, we have 4-level with linear map at 0xFFFF000000000000
            swapper_vaddr = swapper_pg_dir_symbol.address
            if page_table_levels == 3:
                # 3-level can be either 4KB/39-bit or 16KB/47-bit
                # 16KB/47-bit: linear map at 0xFFFFC00000000000
                # 4KB/39-bit: linear map at 0xFFFFFF8000000000
                if (
                    swapper_vaddr >= 0xFFFFC00000000000
                    and swapper_vaddr < 0xFFFFFF8000000000
                ):
                    page_size_kb = 16
                    vollog.debug("Detected 16KB page size (47-bit VA, 3-level)")
                else:
                    page_size_kb = 4
                    vollog.debug("Detected 4KB page size (39-bit VA, 3-level)")
            else:
                # 4-level page tables - currently only 4KB/48-bit supported
                page_size_kb = 4
                vollog.debug("Detected 4KB page size (48-bit VA, 4-level)")

            # Now calculate ASLR with the correct page size
            kaslr_shift, aslr_shift = cls.find_aslr(
                context,
                table_name,
                layer_name,
                page_size_kb,
                progress_callback=progress_callback,
            )

            # Convert virtual to physical for AArch64
            # Use the virtual_to_physical_address method which handles PAGE_OFFSET
            pgd_phys = (
                cls.virtual_to_physical_address(
                    swapper_pg_dir_symbol.address, page_size_kb
                )
                + kaslr_shift
            )

            # Build the new layer
            new_layer_name = context.layers.free_layer_name("AArch64Layer")
            config_path = join("AArch64Helper", new_layer_name)
            context.config[join(config_path, "memory_layer")] = layer_name
            context.config[join(config_path, "page_map_offset")] = pgd_phys
            context.config[join(config_path, "page_table_levels")] = page_table_levels
            context.config[join(config_path, "page_size_kb")] = page_size_kb
            context.config[join(config_path, LinuxSymbolFinder.banner_config_key)] = (
                str(banner, "latin-1")
            )

            layer = arm.LinuxAArch64(
                context,
                config_path=config_path,
                name=new_layer_name,
                metadata={"os": "Linux"},
            )
            layer.config["kernel_virtual_offset"] = aslr_shift

            if layer:
                vollog.debug(f"AArch64 DTB was found at: 0x{pgd_phys:0x}")
                return layer

        vollog.debug("No suitable linux AArch64 banner could be matched")
        return None

    @classmethod
    def find_aslr(
        cls,
        context: interfaces.context.ContextInterface,
        symbol_table: str,
        layer_name: str,
        page_size_kb: int = 4,
        progress_callback: constants.ProgressCallback = None,
    ) -> Tuple[int, int]:
        """Determines the KASLR and ASLR shifts for AArch64."""
        init_task_symbol = symbol_table + constants.BANG + "init_task"
        init_task_json_address = context.symbol_space.get_symbol(
            init_task_symbol
        ).address
        swapper_signature = rb"swapper(\/0|\x00\x00)\x00\x00\x00\x00\x00\x00"
        module = context.module(symbol_table, layer_name, 0)
        address_mask = context.symbol_space[symbol_table].config.get(
            "symbol_mask", None
        )

        task_symbol = module.get_type("task_struct")
        comm_child_offset = task_symbol.relative_child_offset("comm")

        for offset in context.layers[layer_name].scan(
            scanner=scanners.RegExScanner(swapper_signature),
            context=context,
            progress_callback=progress_callback,
        ):
            init_task_address = offset - comm_child_offset
            init_task = module.object(
                object_type="task_struct", offset=init_task_address, absolute=True
            )
            if init_task.pid != 0:
                continue
            elif (
                init_task.has_member("state")
                and init_task.state.cast("unsigned int") != 0
            ):
                continue

            # Calculate ASLR shift
            aslr_shift = (
                int.from_bytes(
                    init_task.files.cast("bytes", length=init_task.files.vol.size),
                    byteorder=init_task.files.vol.data_format.byteorder,
                )
                - module.get_symbol("init_files").address
            )

            # For AArch64, physical addresses are direct mapped
            # Use the correct page size for virtual to physical conversion
            kaslr_shift = init_task_address - cls.virtual_to_physical_address(
                init_task_json_address, page_size_kb
            )

            if address_mask:
                aslr_shift = aslr_shift & address_mask

            if aslr_shift & 0xFFF != 0 or kaslr_shift & 0xFFF != 0:
                continue

            vollog.debug(
                f"Linux AArch64 ASLR shift values determined: physical {kaslr_shift:0x} virtual {aslr_shift:0x}"
            )
            return kaslr_shift, aslr_shift

        vollog.debug("Scanners could not determine any ASLR shifts, using 0 for both")
        return 0, 0

    @staticmethod
    def virtual_to_physical_address(addr: int, page_size_kb: int = 4) -> int:
        """Converts a virtual AArch64 Linux address to a physical one.

        AArch64 Linux kernel virtual addresses in the linear map region
        have the physical address in the lower bits. We subtract the
        PAGE_OFFSET to get the physical address.

        For 16KB pages with 47-bit VA, PAGE_OFFSET is 0xFFFFC00000000000
        For 4KB pages with 48-bit VA, PAGE_OFFSET is 0xFFFF000000000000
        For 4KB pages with 39-bit VA, PAGE_OFFSET is 0xFFFFFF8000000000
        """
        if page_size_kb == 16:
            # 16KB pages, 47-bit VA: PAGE_OFFSET = 0xFFFFC00000000000
            return addr - 0xFFFFC00000000000
        elif addr >= 0xFFFFFF8000000000:
            # 4KB pages, 39-bit VA: PAGE_OFFSET = 0xFFFFFF8000000000
            return addr - 0xFFFFFF8000000000
        else:
            # 4KB pages, 48-bit VA: PAGE_OFFSET = 0xFFFF000000000000
            return addr - 0xFFFF000000000000


class LinuxMIPS64Stacker(interfaces.automagic.StackerLayerInterface):
    """Stacker for Linux MIPS64 memory images."""

    stack_order = 35
    exclusion_list = ["mac", "windows"]

    @classmethod
    def stack(
        cls,
        context: interfaces.context.ContextInterface,
        layer_name: str,
        progress_callback: constants.ProgressCallback = None,
    ) -> Optional[interfaces.layers.DataLayerInterface]:
        """Attempts to identify linux MIPS64 within this layer."""
        layer = context.layers[layer_name]
        join = interfaces.configuration.path_join

        # Never stack on top of an existing translation layer
        if isinstance(layer, (intel.Intel, arm.AArch64, mips.MIPS64)):
            return None

        linux_banners = symbol_cache.load_cache_manager().get_identifier_dictionary(
            operating_system="linux"
        )
        if not linux_banners:
            vollog.info(
                "No Linux banners found - if this is a linux plugin, please check your symbol files location"
            )
            return None

        mss = scanners.MultiStringScanner([x for x in linux_banners if x is not None])
        for _, banner in layer.scan(
            context=context, scanner=mss, progress_callback=progress_callback
        ):
            vollog.debug(f"Identified banner: {repr(banner)}")

            isf_path = linux_banners.get(banner, None)
            if not isf_path:
                continue

            table_name = context.symbol_space.free_table_name("LinuxMIPS64Stacker")
            table = linux.LinuxKernelIntermedSymbols(
                context,
                "temporary." + table_name,
                name=table_name,
                isf_url=isf_path,
            )
            context.symbol_space.append(table)

            # Check if this is a MIPS64 kernel by looking for swapper_pg_dir
            # and checking the address range (MIPS64 CKSEG0 starts at 0xffffffff80000000)
            if "swapper_pg_dir" not in table.symbols:
                vollog.debug("MIPS64 stacker: swapper_pg_dir not found")
                continue

            swapper_pg_dir_symbol = table.get_symbol("swapper_pg_dir")
            swapper_vaddr = swapper_pg_dir_symbol.address
            vollog.debug(f"MIPS64 stacker: swapper_pg_dir at {hex(swapper_vaddr)}")

            # MIPS64 kernel addresses are in CKSEG0 (0xffffffff80000000 - 0xffffffffbfffffff)
            # or KSEG2/3 (0xffffffffc0000000 - 0xffffffffffffffff)
            if not (swapper_vaddr >= 0xFFFFFFFF80000000):
                vollog.debug(
                    f"MIPS64 stacker: address {hex(swapper_vaddr)} not in MIPS64 range"
                )
                continue

            vollog.debug(f"Detected MIPS64 kernel at {hex(swapper_vaddr)}")

            # Convert virtual to physical for MIPS64
            # CKSEG0: 0xffffffff8xxxxxxx -> physical 0x0xxxxxxx (lower 29 bits)
            pgd_phys = cls.virtual_to_physical_address(swapper_vaddr)

            # Build the new layer
            new_layer_name = context.layers.free_layer_name("MIPS64Layer")
            config_path = join("MIPS64Helper", new_layer_name)
            context.config[join(config_path, "memory_layer")] = layer_name
            context.config[join(config_path, "page_map_offset")] = pgd_phys
            context.config[join(config_path, LinuxSymbolFinder.banner_config_key)] = (
                str(banner, "latin-1")
            )

            layer = mips.LinuxMIPS64(
                context,
                config_path=config_path,
                name=new_layer_name,
                metadata={"os": "Linux"},
            )

            # MIPS64 doesn't use KASLR, set kernel_virtual_offset to 0
            layer.config["kernel_virtual_offset"] = 0

            if layer:
                vollog.debug(f"MIPS64 DTB was found at: 0x{pgd_phys:0x}")
                return layer

        vollog.debug("No suitable linux MIPS64 banner could be matched")
        return None

    @staticmethod
    def virtual_to_physical_address(addr: int) -> int:
        """Converts a virtual MIPS64 Linux address to a physical one.

        MIPS64 memory segments:
        - CKSEG0 (0xffffffff80000000 - 0xffffffff9fffffff): physical = vaddr & 0x1fffffff
        - CKSEG1 (0xffffffffa0000000 - 0xffffffffbfffffff): physical = vaddr & 0x1fffffff
        - XKPHYS (0x8000000000000000 - 0xbfffffffffffffff): physical = vaddr & 0x07ffffffffffffff
        """
        # CKSEG0/CKSEG1: lower 29 bits are physical address
        if addr >= 0xFFFFFFFF80000000 and addr < 0xFFFFFFFFC0000000:
            return addr & 0x1FFFFFFF
        # XKPHYS: lower 59 bits are physical address
        if addr >= 0x8000000000000000 and addr < 0xC000000000000000:
            return addr & 0x07FFFFFFFFFFFFFF
        # For mapped addresses, this is a fallback
        return addr & 0xFFFFFFFF


class LinuxPPC32Stacker(interfaces.automagic.StackerLayerInterface):
    """Stacker for Linux PPC32 (PowerPC 32-bit) memory images."""

    stack_order = 35
    exclusion_list = ["mac", "windows"]

    @classmethod
    def stack(
        cls,
        context: interfaces.context.ContextInterface,
        layer_name: str,
        progress_callback: constants.ProgressCallback = None,
    ) -> Optional[interfaces.layers.DataLayerInterface]:
        """Attempts to identify linux PPC32 within this layer."""
        layer = context.layers[layer_name]
        join = interfaces.configuration.path_join

        # Never stack on top of an existing translation layer
        if isinstance(layer, (intel.Intel, arm.AArch64, mips.MIPS64, ppc.PPC32)):
            return None

        linux_banners = symbol_cache.load_cache_manager().get_identifier_dictionary(
            operating_system="linux"
        )
        if not linux_banners:
            vollog.info(
                "No Linux banners found - if this is a linux plugin, please check your symbol files location"
            )
            return None

        mss = scanners.MultiStringScanner([x for x in linux_banners if x is not None])
        for banner_offset, banner in layer.scan(
            context=context, scanner=mss, progress_callback=progress_callback
        ):
            vollog.debug(f"Identified banner: {repr(banner)}")

            isf_path = linux_banners.get(banner, None)
            if not isf_path:
                continue

            table_name = context.symbol_space.free_table_name("LinuxPPC32Stacker")
            table = linux.LinuxKernelIntermedSymbols(
                context,
                "temporary." + table_name,
                name=table_name,
                isf_url=isf_path,
            )
            context.symbol_space.append(table)

            # Check if this is a PPC32 kernel by examining pointer size and address range
            # PPC32 uses 32-bit pointers and PAGE_OFFSET is typically 0xc0000000
            if "swapper_pg_dir" not in table.symbols:
                vollog.debug("PPC32 stacker: swapper_pg_dir not found")
                continue

            swapper_pg_dir_symbol = table.get_symbol("swapper_pg_dir")
            swapper_vaddr = swapper_pg_dir_symbol.address

            # PPC32 kernel addresses are in the range 0xc0000000 - 0xffffffff
            # and should be 32-bit (not 64-bit like MIPS64 or AArch64)
            if swapper_vaddr >= 0x100000000:
                vollog.debug(
                    f"PPC32 stacker: address {hex(swapper_vaddr)} is 64-bit, skipping"
                )
                continue

            if swapper_vaddr < 0xC0000000:
                vollog.debug(
                    f"PPC32 stacker: address {hex(swapper_vaddr)} not in PPC32 kernel range"
                )
                continue

            # Check for PPC-specific symbols to confirm architecture
            # PPC kernels have machine_check_exception, not idt_table (x86)
            if "idt_table" in table.symbols:
                vollog.debug("PPC32 stacker: idt_table found, this is x86, skipping")
                continue

            vollog.debug(f"Detected PPC32 kernel at {hex(swapper_vaddr)}")

            # Get linux_banner symbol to calculate physical offset
            if "linux_banner" not in table.symbols:
                vollog.debug("PPC32 stacker: linux_banner symbol not found")
                continue

            linux_banner_vaddr = table.get_symbol("linux_banner").address

            # Calculate physical offset: banner_phys = banner_vaddr - PAGE_OFFSET + phys_offset
            # We know banner_phys (from scan) and banner_vaddr (from symbol table)
            # PAGE_OFFSET is typically 0xc0000000 for PPC32
            # So: phys_offset = banner_phys - (banner_vaddr - PAGE_OFFSET)
            #                 = banner_phys - banner_vaddr + PAGE_OFFSET
            page_offset = 0xC0000000
            phys_offset = banner_offset - (linux_banner_vaddr - page_offset)

            vollog.debug(
                f"PPC32: banner_phys={hex(banner_offset)}, banner_vaddr={hex(linux_banner_vaddr)}, phys_offset={hex(phys_offset)}"
            )

            # Build the new layer
            new_layer_name = context.layers.free_layer_name("PPC32Layer")
            config_path = join("PPC32Helper", new_layer_name)
            context.config[join(config_path, "memory_layer")] = layer_name
            context.config[join(config_path, "kernel_virtual_offset")] = page_offset
            context.config[join(config_path, "kernel_physical_offset")] = phys_offset
            context.config[join(config_path, LinuxSymbolFinder.banner_config_key)] = (
                str(banner, "latin-1")
            )

            # Get vmalloc-related symbols for page-based translation
            vmap_area_list = 0
            mem_map = 0
            page_struct_size = 36  # Default for PPC32

            if "vmap_area_list" in table.symbols:
                vmap_area_list = table.get_symbol("vmap_area_list").address
                vollog.debug(f"PPC32: vmap_area_list at {hex(vmap_area_list)}")

            if "mem_map" in table.symbols:
                mem_map = table.get_symbol("mem_map").address
                vollog.debug(f"PPC32: mem_map at {hex(mem_map)}")

            # Get struct page size from symbol table
            if "page" in table.types:
                page_type = table.get_type("page")
                page_struct_size = page_type.size
                vollog.debug(f"PPC32: struct page size = {page_struct_size}")

            context.config[join(config_path, "vmap_area_list")] = vmap_area_list
            context.config[join(config_path, "mem_map")] = mem_map
            context.config[join(config_path, "page_struct_size")] = page_struct_size

            new_layer = ppc.PPC32(
                context,
                config_path=config_path,
                name=new_layer_name,
                metadata={"os": "Linux"},
            )

            # PPC32 symbols already include PAGE_OFFSET, so module offset is 0
            new_layer.config["kernel_virtual_offset"] = 0

            if new_layer:
                vollog.debug(
                    f"PPC32 layer created with PAGE_OFFSET={hex(page_offset)}, phys_offset={hex(phys_offset)}"
                )
                return new_layer

        vollog.debug("No suitable linux PPC32 banner could be matched")
        return None


class LinuxSymbolFinder(symbol_finder.SymbolFinder):
    """Linux symbol loader based on uname signature strings."""

    banner_config_key = "kernel_banner"
    operating_system = "linux"
    symbol_class = "volatility3.framework.symbols.linux.LinuxKernelIntermedSymbols"
    exclusion_list = ["mac", "windows"]

    @classmethod
    def find_aslr(cls, *args):
        return LinuxIntelStacker.find_aslr(*args)[1]


class LinuxIntelVMCOREINFOStacker(interfaces.automagic.StackerLayerInterface):
    stack_order = 34
    exclusion_list = ["mac", "windows"]

    @staticmethod
    def _check_versions() -> bool:
        """Verify the versions of the required modules"""
        # Check VMCOREINFO API version
        vmcoreinfo_version_required = (1, 0, 0)
        if not requirements.VersionRequirement.matches_required(
            vmcoreinfo_version_required, linux.VMCoreInfo.version
        ):
            vollog.info(
                "VMCOREINFO version not suitable: required %s found %s",
                vmcoreinfo_version_required,
                linux.VMCoreInfo.version,
            )
            return False

        return True

    @classmethod
    def stack(
        cls,
        context: interfaces.context.ContextInterface,
        layer_name: str,
        progress_callback: constants.ProgressCallback = None,
    ) -> Optional[interfaces.layers.DataLayerInterface]:
        """Attempts to identify linux within this layer."""

        # Verify the versions of the required modules
        if not cls._check_versions():
            return None

        # Bail out by default unless we can stack properly
        layer = context.layers[layer_name]

        # Never stack on top of a translation layer (Intel, AArch64, MIPS64, or PPC32)
        if isinstance(layer, (intel.Intel, arm.AArch64, mips.MIPS64, ppc.PPC32)):
            return None

        linux_banners = symbol_cache.load_cache_manager().get_identifier_dictionary(
            operating_system="linux"
        )
        if not linux_banners:
            # If we have no banners, don't bother scanning
            vollog.info(
                "No Linux banners found - if this is a linux plugin, please check your "
                "symbol files location"
            )
            return None

        vmcoreinfo_elf_notes_iter = linux.VMCoreInfo.search_vmcoreinfo_elf_note(
            context=context,
            layer_name=layer_name,
            progress_callback=progress_callback,
        )

        # Iterate through each VMCOREINFO ELF note found, using the first one that is valid.
        for _vmcoreinfo_offset, vmcoreinfo in vmcoreinfo_elf_notes_iter:
            shifts = cls._vmcoreinfo_find_aslr(vmcoreinfo)
            if not shifts:
                # Let's try the next VMCOREINFO, in case this one isn't correct.
                continue

            kaslr_shift, aslr_shift = shifts

            dtb = cls._vmcoreinfo_get_dtb(vmcoreinfo, aslr_shift, kaslr_shift)
            if dtb is None:
                # Discard this VMCOREINFO immediately
                continue

            is_32bit, is_pae = cls._vmcoreinfo_is_32bit(vmcoreinfo)
            if is_32bit:
                layer_class = intel.IntelPAE if is_pae else intel.Intel
            else:
                layer_class = intel.Intel32e

            uts_release = vmcoreinfo["OSRELEASE"]

            # See how linux_banner constant is built in the linux kernel
            linux_version_prefix = f"Linux version {uts_release} (".encode()
            valid_banners = [
                x for x in linux_banners if x and x.startswith(linux_version_prefix)
            ]
            if not valid_banners:
                # There's no banner matching this VMCOREINFO, keep trying with the next one
                continue
            elif len(valid_banners) == 1:
                # Usually, we narrow down the Linux banner list to a single element.
                # Using BytesScanner here is slightly faster than MultiStringScanner.
                scanner = scanners.BytesScanner(valid_banners[0])
            else:
                scanner = scanners.MultiStringScanner(valid_banners)

            join = interfaces.configuration.path_join
            for match in layer.scan(
                context=context, scanner=scanner, progress_callback=progress_callback
            ):
                # Unfortunately, the scanners do not maintain a consistent interface
                banner = match[1] if isinstance(match, Tuple) else valid_banners[0]

                isf_path = linux_banners.get(banner, None)
                if not isf_path:
                    vollog.warning(
                        "Identified banner %r, but no matching ISF is available.",
                        banner,
                    )
                    continue

                vollog.debug("Identified banner: %r", banner)
                table_name = context.symbol_space.free_table_name("LintelStacker")
                table = linux.LinuxKernelIntermedSymbols(
                    context,
                    f"temporary.{table_name}",
                    name=table_name,
                    isf_url=isf_path,
                )
                context.symbol_space.append(table)

                # Build the new layer
                new_layer_name = context.layers.free_layer_name("primary")
                config_path = join("vmcoreinfo", new_layer_name)
                kernel_banner = LinuxSymbolFinder.banner_config_key
                banner_str = banner.decode(encoding="latin-1")
                context.config[join(config_path, kernel_banner)] = banner_str
                context.config[join(config_path, "memory_layer")] = layer_name
                context.config[join(config_path, "page_map_offset")] = dtb
                context.config[join(config_path, "kernel_virtual_offset")] = aslr_shift
                layer = layer_class(
                    context,
                    config_path=config_path,
                    name=new_layer_name,
                    metadata={"os": "Linux"},
                )

                if layer:
                    vollog.debug(
                        "Values found in VMCOREINFO: KASLR=0x%x, ASLR=0x%x, DTB=0x%x",
                        kaslr_shift,
                        aslr_shift,
                        dtb,
                    )

                    return layer

        vollog.debug("No suitable linux banner could be matched")
        return None

    @staticmethod
    def _vmcoreinfo_find_aslr(vmcoreinfo) -> Tuple[int, int]:
        phys_base = vmcoreinfo.get("NUMBER(phys_base)")
        if phys_base is None:
            # In kernel < 4.10, there may be a SYMBOL(phys_base), but as noted in the
            # c401721ecd1dcb0a428aa5d6832ee05ffbdbffbbe commit comment, this value
            # isn't useful for calculating the physical address.
            # There's nothing we can do here, so let's try with the next VMCOREINFO or
            # the next Stacker.
            return None

        # kernels 3.14 (b6085a865762236bb84934161273cdac6dd11c2d) KERNELOFFSET was added
        kerneloffset = vmcoreinfo.get("KERNELOFFSET")
        if kerneloffset is None:
            # kernels < 3.14 if KERNELOFFSET is missing, KASLR might not be implemented.
            # Oddly, NUMBER(phys_base) is present without it. To be safe, proceed only
            # if both are present.
            return None

        aslr_shift = kerneloffset
        kaslr_shift = phys_base + aslr_shift

        return kaslr_shift, aslr_shift

    @staticmethod
    def _vmcoreinfo_get_dtb(vmcoreinfo, aslr_shift, kaslr_shift) -> int:
        """Returns the page global directory physical address (a.k.a DTB or PGD)"""
        # In x86-64, since kernels 2.5.22 swapper_pg_dir is a macro to the respective pgd.
        # First, in e3ebadd95cb621e2c7436f3d3646447ac9d5c16d to init_level4_pgt, and later
        # in kernels 4.13 in 65ade2f872b474fa8a04c2d397783350326634e6) to init_top_pgt.
        # In x86-32, the pgd is swapper_pg_dir. So, in any case, for VMCOREINFO
        # SYMBOL(swapper_pg_dir) will always have the right value.
        dtb_vaddr = vmcoreinfo.get("SYMBOL(swapper_pg_dir)")
        if dtb_vaddr is None:
            # Abort, it should be present
            return None

        dtb_paddr = (
            LinuxIntelStacker.virtual_to_physical_address(dtb_vaddr)
            - aslr_shift
            + kaslr_shift
        )

        return dtb_paddr

    @staticmethod
    def _vmcoreinfo_is_32bit(vmcoreinfo) -> Tuple[bool, bool]:
        """Returns a tuple of booleans with is_32bit and is_pae values"""
        is_pae = vmcoreinfo.get("CONFIG_X86_PAE", "n") == "y"
        if is_pae:
            is_32bit = True
        else:
            # Check the swapper_pg_dir virtual address size
            dtb_vaddr = vmcoreinfo["SYMBOL(swapper_pg_dir)"]
            is_32bit = dtb_vaddr <= 2**32

        return is_32bit, is_pae
