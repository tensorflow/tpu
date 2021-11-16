/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/pci.h>
#include "drivers/asic_sw/asic_fw_state_offsets.h"
#include "drivers/asic_sw/asic_fw_reinit_reset_offsets.h"
#include "drivers/char/tpu_common/tpu_common.h"
#include "drivers/char/tpu_common/tpuv4.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_bar0_ranges.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_bar2_ranges.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_device_owner_offsets.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_mgr_error_loperf_offsets.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_firmware_state_offsets.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_firmware_version_offsets.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_interrupt_desc.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_userspace_firmware_indirect_accessor_offsets.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_userspace_lst_port_indirect_offsets.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_page_table_config.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_pcie_flr_status_offsets.h"
#include "drivers/char/tpu_common/tpuv4/tpuv4_reinit_reset_offsets.h"
#include "drivers/char/tpu_common/tpuv4common.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_mgr_error_loperf_offsets.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_pcie_flr_status_offsets.h"
#include "drivers/gasket/gasket_core.h"
#include "drivers/gasket/gasket_logging.h"
#define TPUV4_PCI_DEVICE_ID 0x005e
#define TPUV4_PCI_VENDOR_ID 0x1ae0
static const struct pci_device_id tpuv4_pci_ids[] = {
 { PCI_DEVICE(TPUV4_PCI_VENDOR_ID, TPUV4_PCI_DEVICE_ID) },
 { 0 },
};
static const struct gasket_mappable_region lbus_bar_regions[] = {
 { 0x0, TPUV4_LBUS_BAR_BYTES },
};
static const struct gasket_mappable_region csr_bar_regions[] = {
 { 0x0, TPUV4_CSR_BAR_BYTES },
};
static const struct tpuv4_setup_data setup_data = {
 .software_csr_bar_index = TPUV4_CSR_BAR_INDEX,
 .reinit_reset_value = TPUV4_CHIP_REINIT_RESET,
 .reset_accepted_value = TPUV4_RESET_ACCEPTED,
 .chip_init_done_value = TPUV4_CHIP_INIT_DONE,
 .reset_retry_value = TPUV4_RESET_RETRY,
 .reset_delay_value = TPUV4_RESET_DELAY,
 .error_offsets_get = tpuv4_mgr_error_loperf_offsets_get,
 .device_owner_offsets_get = tpuv4_device_owner_offsets_get,
 .indirect_register_offsets_get =
  tpuv4_userspace_firmware_indirect_accessor_offsets_get,
 .userspace_lst_port_indirect_offsets_count =
  tpuv4_userspace_lst_port_indirect_offsets_count,
 .userspace_lst_port_indirect_offsets_get =
  tpuv4_userspace_lst_port_indirect_offsets_get,
 .bar0_get_region_count = tpuv4_bar0_get_region_count,
 .bar0_get_regions = tpuv4_bar0_get_regions,
 .bar2_get_region_count = tpuv4_bar2_get_region_count,
 .bar2_get_regions = tpuv4_bar2_get_regions,
 .firmware_state_offsets_get = tpuv4_firmware_state_offsets_get,
 .firmware_version_offsets_get = tpuv4_firmware_version_offsets_get,
 .firmware_reinit_reset_offsets_get = tpuv4_reinit_reset_offsets_get,
 .pcie_flr_status_offsets_get = tpuv4_pcie_flr_status_offsets_get,
};
static int add_dev_cb(struct gasket_dev *gasket_dev)
{
 return tpuv4_initialize(gasket_dev, &setup_data);
}
static struct gasket_driver_desc tpuv4_driver_desc = {
 .driver_version = TPUV4_DRIVER_VERSION,
 .num_page_tables = ARRAY_SIZE(tpuv4_page_table_configs),
 .page_table_configs = tpuv4_page_table_configs,
 .bar_descriptions = {
  { TPUV4_LBUS_BAR_BYTES, VM_READ, TPUV4_LBUS_BAR_OFFSET,
   ARRAY_SIZE(lbus_bar_regions), lbus_bar_regions },
  GASKET_UNUSED_BAR,
  { TPUV4_CSR_BAR_BYTES, VM_READ | VM_WRITE, TPUV4_CSR_BAR_OFFSET,
   ARRAY_SIZE(csr_bar_regions), csr_bar_regions},
  GASKET_UNUSED_BAR,
  GASKET_UNUSED_BAR,
  GASKET_UNUSED_BAR,
 },
 .legacy_interrupt_bar_index = 0,
 .num_msix_interrupts = TPUV4_NUM_MSIX_INTERRUPTS,
 .num_interrupts = ARRAY_SIZE(tpuv4_interrupts),
 .interrupts = tpuv4_interrupts,
 .legacy_interrupts = NULL,
 .add_dev_cb = add_dev_cb,
 .remove_dev_cb = tpuv4_remove_dev_cb,
 .enable_dev_cb = NULL,
 .disable_dev_cb = NULL,
 .sysfs_setup_cb = tpu_common_sysfs_setup,
 .sysfs_cleanup_cb = NULL,
 .device_open_cb = tpuv4_device_open_cb,
 .device_release_cb = NULL,
 .device_close_cb = tpuv4_device_close_cb,
 .get_mappable_regions_cb = tpuv4_get_mappable_regions_cb,
 .ioctl_handler_cb = NULL,
 .device_status_cb = tpuv4_get_status,
 .hardware_revision_cb = tpu_common_get_hardware_revision,
 .firmware_version_cb = tpu_common_get_firmware_version_cb,
 .device_reset_cb = tpuv4_reset,
};
static void tpuv4_exit(void);
static int __init tpuv4_init(void);
static struct gasket_device_desc device_desc = {
 .name = TPU_COMMON_ACCEL_TYPE,
 .legacy_support = 0,
 .module = THIS_MODULE,
 .pci_id_table = tpuv4_pci_ids,
 .driver_desc = &tpuv4_driver_desc,
};
MODULE_DESCRIPTION("Google tpu_v4 driver");
MODULE_VERSION(TPUV4_DRIVER_VERSION);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("asic-sw <noreply@google.com>");
MODULE_DEVICE_TABLE(pci, tpuv4_pci_ids);
module_init(tpuv4_init);
module_exit(tpuv4_exit);
int __init tpuv4_init(void)
{
 return gasket_register_device(&device_desc);
}
void tpuv4_exit(void)
{
 gasket_unregister_device(&device_desc);
}
