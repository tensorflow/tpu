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
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_bar0_ranges.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_bar2_ranges.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_mgr_error_loperf_offsets.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_device_owner_offsets.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_firmware_state_offsets.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_firmware_version_offsets.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_interrupt_desc.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_userspace_firmware_indirect_accessor_offsets.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_userspace_lst_port_indirect_offsets.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_page_table_config.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_pcie_flr_status_offsets.h"
#include "drivers/char/tpu_common/tpuv4lite/tpuv4lite_reinit_reset_offsets.h"
#include "drivers/char/tpu_common/tpuv4common.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_mgr_error_loperf_offsets.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_pcie_flr_status_offsets.h"
#include "drivers/gasket/gasket_core.h"
#include "drivers/gasket/gasket_logging.h"
#define TPUV4LITE_DRIVER_VERSION "0.0.1"
#define TPUV4LITE_PCI_DEVICE_ID 0x0056
#define TPUV4LITE_PCI_VENDOR_ID 0x1ae0
#define TPUV4LITE_NUM_MSIX_INTERRUPTS 128
static void tpuv4lite_exit(void);
static int __init tpuv4lite_init(void);
static int add_dev_cb(struct gasket_dev *gasket_dev);
static int remove_dev_cb(struct gasket_dev *gasket_dev);
static int
tpuv4lite_get_mappable_regions_cb(
 struct gasket_filp_data *filp_data, int bar_index,
 struct gasket_mappable_region **mappable_regions,
 int *num_mappable_regions);
static int tpuv4lite_reset(struct gasket_dev *gasket_dev, uint type);
struct tpuv4lite_device_data {
 struct tpuv4common_device_data tpuv4common_data;
};
static int tpuv4lite_device_open_cb(
 struct gasket_filp_data *filp_data, struct file *file)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct tpuv4lite_device_data *device_data = gasket_dev->cb_data;
 return tpuv4common_device_open(gasket_dev, &(device_data->tpuv4common_data));
}
int tpuv4lite_device_close_cb(struct gasket_filp_data *filp_data, struct file *file)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct tpuv4lite_device_data *tpuv4lite_data = gasket_dev->cb_data;
 const struct tpuv4common_mgr_error_loperf_offsets *mgr_error_loperf_offsets;
 unsigned long global_fatal_error_status_offset;
 mgr_error_loperf_offsets = tpuv4lite_mgr_error_loperf_offsets_get(0);
 global_fatal_error_status_offset =
  mgr_error_loperf_offsets->global_fatal_error_status;
 return tpuv4common_device_close(gasket_dev, &(tpuv4lite_data->tpuv4common_data),
    TPUV4LITE_CSR_BAR_INDEX,
    global_fatal_error_status_offset,
    TPUV4LITE_CHIP_REINIT_RESET);
}
enum gasket_status tpuv4lite_get_status(struct gasket_dev *gasket_dev)
{
 return GASKET_STATUS_ALIVE;
}
static const struct pci_device_id tpuv4lite_pci_ids[] = {
 { PCI_DEVICE(TPUV4LITE_PCI_VENDOR_ID, TPUV4LITE_PCI_DEVICE_ID) },
 { 0 },
};
static const struct gasket_mappable_region lbus_bar_regions[] = {
 { 0x0, TPUV4LITE_LBUS_BAR_BYTES },
};
static const struct gasket_mappable_region csr_bar_regions[] = {
 { 0x0, TPUV4LITE_CSR_BAR_BYTES },
};
static struct gasket_driver_desc tpuv4lite_driver_desc = {
 .chip_version = "1.0.0",
 .driver_version = TPUV4LITE_DRIVER_VERSION,
 .num_page_tables = ARRAY_SIZE(tpuv4lite_page_table_configs),
 .page_table_configs = tpuv4lite_page_table_configs,
 .bar_descriptions = {
  { TPUV4LITE_LBUS_BAR_BYTES, VM_READ, TPUV4LITE_LBUS_BAR_OFFSET,
   ARRAY_SIZE(lbus_bar_regions), lbus_bar_regions },
  GASKET_UNUSED_BAR,
  { TPUV4LITE_CSR_BAR_BYTES, VM_READ | VM_WRITE, TPUV4LITE_CSR_BAR_OFFSET,
   ARRAY_SIZE(csr_bar_regions), csr_bar_regions},
  GASKET_UNUSED_BAR,
  GASKET_UNUSED_BAR,
  GASKET_UNUSED_BAR,
 },
 .legacy_interrupt_bar_index = 0,
 .num_msix_interrupts = TPUV4LITE_NUM_MSIX_INTERRUPTS,
 .num_interrupts = ARRAY_SIZE(tpuv4lite_interrupts),
 .interrupts = tpuv4lite_interrupts,
 .legacy_interrupts = NULL,
 .add_dev_cb = add_dev_cb,
 .remove_dev_cb = remove_dev_cb,
 .enable_dev_cb = NULL,
 .disable_dev_cb = NULL,
 .sysfs_setup_cb = tpu_common_sysfs_setup,
 .sysfs_cleanup_cb = NULL,
 .device_open_cb = tpuv4lite_device_open_cb,
 .device_release_cb = NULL,
 .device_close_cb = tpuv4lite_device_close_cb,
 .get_mappable_regions_cb = tpuv4lite_get_mappable_regions_cb,
 .ioctl_handler_cb = NULL,
 .device_status_cb = tpuv4lite_get_status,
 .hardware_revision_cb = tpu_common_get_hardware_revision,
 .firmware_version_cb = tpu_common_get_firmware_version_cb,
 .device_reset_cb = tpuv4lite_reset,
};
static struct gasket_device_desc device_desc = {
 .name = TPU_COMMON_ACCEL_TYPE,
 .legacy_support = 0,
 .module = THIS_MODULE,
 .pci_id_table = tpuv4lite_pci_ids,
 .driver_desc = &tpuv4lite_driver_desc,
};
MODULE_DESCRIPTION("Google tpu_v4_lite driver");
MODULE_VERSION(TPUV4LITE_DRIVER_VERSION);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Googler <noreply@google.com>");
MODULE_DEVICE_TABLE(pci, tpuv4lite_pci_ids);
module_init(tpuv4lite_init);
module_exit(tpuv4lite_exit);
int __init tpuv4lite_init(void)
{
 return gasket_register_device(&device_desc);
}
void tpuv4lite_exit(void)
{
 gasket_unregister_device(&device_desc);
}
static int add_dev_cb(struct gasket_dev *gasket_dev)
{
 int ret = 0;
 struct tpuv4lite_device_data *tpuv4lite_device =
  kzalloc(sizeof(struct tpuv4lite_device_data), GFP_KERNEL);
 if (!tpuv4lite_device) {
  gasket_log_error(
   gasket_dev,
   "Unable to initialize tpu_v4_lite device storage.");
  return -ENOMEM;
 }
 ret = tpuv4common_setup_device_data(
  &(tpuv4lite_device->tpuv4common_data), TPUV4LITE_CHIP_REINIT_RESET,
                       TPUV4LITE_CSR_BAR_INDEX,
  tpuv4lite_device_owner_offsets_get(0)->device_owner,
  tpuv4lite_firmware_version_offsets_get(0)->primary_version,
  tpuv4lite_userspace_firmware_indirect_accessor_offsets_get(0),
  tpuv4lite_userspace_lst_port_indirect_offsets_count(),
  tpuv4lite_userspace_lst_port_indirect_offsets_get);
 if (ret) {
  gasket_log_error(gasket_dev, "Failed to setup device_data");
  goto setup_failed;
 }
 gasket_dev->cb_data = tpuv4lite_device;
 return 0;
setup_failed:
 kfree(tpuv4lite_device);
 return ret;
}
static int remove_dev_cb(struct gasket_dev *gasket_dev)
{
 if (gasket_dev->cb_data == NULL)
  return -EINVAL;
 kfree(gasket_dev->cb_data);
 gasket_dev->cb_data = NULL;
 return 0;
}
static int tpuv4lite_get_bar_region_count(struct gasket_dev *gasket_dev, int bar, enum tpu_common_security_level group)
{
 switch (bar) {
 case 0:
  return tpuv4lite_bar0_get_region_count(group);
 case 2:
  return tpuv4lite_bar2_get_region_count(group);
 default:
  return 0;
 }
}
static const struct gasket_mappable_region *
tpuv4lite_get_bar_regions(struct gasket_dev *gasket_dev, int bar, enum tpu_common_security_level group)
{
 switch (bar) {
 case 0:
  return tpuv4lite_bar0_get_regions(group);
 case 2:
  return tpuv4lite_bar2_get_regions(group);
 default:
  return NULL;
 }
}
int tpuv4lite_get_mappable_regions_cb(
 struct gasket_filp_data *filp_data, int bar_index,
 struct gasket_mappable_region **mappable_regions,
 int *num_mappable_regions)
{
 return tpu_common_get_mappable_regions(
  filp_data->gasket_dev, bar_index, tpuv4lite_get_bar_region_count,
  tpuv4lite_get_bar_regions, mappable_regions, num_mappable_regions);
}
static int reset_complete(struct gasket_dev *gasket_dev, bool log_not_complete)
{
 const struct asic_fw_state_offsets *state_offsets;
 const struct asic_fw_reinit_reset_offsets *reinit_reset_offsets;
 const struct tpuv4common_pcie_flr_status_offsets *pcie_flr_offsets;
 state_offsets = tpuv4lite_firmware_state_offsets_get(0);
 reinit_reset_offsets = tpuv4lite_reinit_reset_offsets_get(0);
 pcie_flr_offsets = tpuv4lite_pcie_flr_status_offsets_get(0);
 return tpuv4common_reset_complete(
  gasket_dev, TPUV4LITE_CSR_BAR_INDEX, reinit_reset_offsets->chip_reset,
  TPUV4LITE_RESET_ACCEPTED, state_offsets->firmware_status,
  TPUV4LITE_CHIP_INIT_DONE, pcie_flr_offsets->pcie_flr_status,
                          0, log_not_complete);
}
int tpuv4lite_reset(struct gasket_dev *gasket_dev, uint type)
{
 const struct asic_fw_state_offsets *state_offsets;
 const struct asic_fw_reinit_reset_offsets *reinit_reset_offsets;
 if (type == TPUV4LITE_CHIP_REINIT_RESET) {
  state_offsets = tpuv4lite_firmware_state_offsets_get(0);
  reinit_reset_offsets = tpuv4lite_reinit_reset_offsets_get(0);
  return tpu_common_reinit_reset(gasket_dev, TPUV4LITE_CSR_BAR_INDEX,
         TPUV4LITE_RESET_RETRY, TPUV4LITE_RESET_DELAY,
         reset_complete,
         reinit_reset_offsets->chip_reset,
         TPUV4LITE_CHIP_REINIT_RESET,
         TPUV4LITE_RESET_ACCEPTED);
 } else {
  gasket_log_error(gasket_dev, "invalid reset type specified: %u",
     type);
  return -EINVAL;
 }
}
