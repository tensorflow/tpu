/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "drivers/char/tpu_common/tpuv4.h"
#include "drivers/gasket/gasket_logging.h"
struct tpuv4_device_data {
 struct tpuv4common_device_data tpuv4common_data;
 int software_csr_bar_index;
 int reinit_reset_value;
 int reset_accepted_value;
 int chip_init_done_value;
 int reset_retry_value;
 int reset_delay_value;
 uint64 global_fatal_error_status_offset;
 int (*bar0_get_region_count)(enum tpu_common_security_level group);
 const struct gasket_mappable_region *(*bar0_get_regions)(
  enum tpu_common_security_level group);
 int (*bar2_get_region_count)(enum tpu_common_security_level group);
 const struct gasket_mappable_region *(*bar2_get_regions)(
  enum tpu_common_security_level group);
 const struct asic_fw_state_offsets *firmware_state_offsets;
 const struct asic_fw_version_offsets *firmware_version_offsets;
 const struct asic_fw_reinit_reset_offsets *firmware_reinit_reset_offsets;
 const struct tpuv4common_pcie_flr_status_offsets *pcie_flr_status_offsets;
};
int tpuv4_device_open_cb(struct gasket_filp_data *filp_data, struct file *file)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct tpuv4_device_data *device_data = gasket_dev->cb_data;
 return tpuv4common_device_open(gasket_dev, &(device_data->tpuv4common_data));
}
int tpuv4_device_close_cb(struct gasket_filp_data *filp_data, struct file *file)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct tpuv4_device_data *tpuv4_data = gasket_dev->cb_data;
 return tpuv4common_device_close(gasket_dev, &(tpuv4_data->tpuv4common_data),
    tpuv4_data->software_csr_bar_index,
    tpuv4_data->global_fatal_error_status_offset,
    tpuv4_data->reinit_reset_value);
}
enum gasket_status tpuv4_get_status(struct gasket_dev *gasket_dev)
{
 return GASKET_STATUS_ALIVE;
}
int tpuv4_initialize(struct gasket_dev *gasket_dev,
     const struct tpuv4_setup_data *setup_data)
{
 const struct tpuv4common_mgr_error_loperf_offsets *error_offsets;
 const struct asic_fw_device_owner_offsets *device_owner_offsets;
 int ret = 0;
 struct tpuv4_device_data *tpuv4_data =
  kzalloc(sizeof(struct tpuv4_device_data), GFP_KERNEL);
 if (!tpuv4_data) {
  gasket_log_error(
   gasket_dev,
   "Unable to initialize tpu_v4 device storage.");
  return -ENOMEM;
 }
 tpuv4_data->software_csr_bar_index = setup_data->software_csr_bar_index;
 tpuv4_data->reinit_reset_value = setup_data->reinit_reset_value;
 tpuv4_data->reset_accepted_value = setup_data->reset_accepted_value;
 tpuv4_data->chip_init_done_value = setup_data->chip_init_done_value;
 tpuv4_data->reset_retry_value = setup_data->reset_retry_value;
 tpuv4_data->reset_delay_value = setup_data->reset_delay_value;
 error_offsets = setup_data->error_offsets_get(0);
 tpuv4_data->global_fatal_error_status_offset =
  error_offsets->global_fatal_error_status;
 tpuv4_data->bar0_get_region_count = setup_data->bar0_get_region_count;
 tpuv4_data->bar0_get_regions = setup_data->bar0_get_regions;
 tpuv4_data->bar2_get_region_count = setup_data->bar2_get_region_count;
 tpuv4_data->bar2_get_regions = setup_data->bar2_get_regions;
 tpuv4_data->firmware_state_offsets =
  setup_data->firmware_state_offsets_get(0);
 tpuv4_data->firmware_version_offsets =
  setup_data->firmware_version_offsets_get(0);
 tpuv4_data->firmware_reinit_reset_offsets =
  setup_data->firmware_reinit_reset_offsets_get(0);
 tpuv4_data->pcie_flr_status_offsets =
  setup_data->pcie_flr_status_offsets_get(0);
 device_owner_offsets = setup_data->device_owner_offsets_get(0);
 ret = tpuv4common_setup_device_data(
  &(tpuv4_data->tpuv4common_data), tpuv4_data->reinit_reset_value,
                       tpuv4_data->software_csr_bar_index,
  device_owner_offsets->device_owner,
  tpuv4_data->firmware_version_offsets->primary_version,
  setup_data->indirect_register_offsets_get(0),
  setup_data->userspace_lst_port_indirect_offsets_count(),
  setup_data->userspace_lst_port_indirect_offsets_get);
 if (ret) {
  gasket_log_error(gasket_dev, "Failed to setup device_data");
  goto setup_failed;
 }
 gasket_dev->cb_data = tpuv4_data;
 return 0;
setup_failed:
 kfree(tpuv4_data);
 return ret;
}
int tpuv4_remove_dev_cb(struct gasket_dev *gasket_dev)
{
 if (gasket_dev->cb_data == NULL)
  return -EINVAL;
 kfree(gasket_dev->cb_data);
 gasket_dev->cb_data = NULL;
 return 0;
}
static int tpuv4_get_bar_region_count(struct gasket_dev *gasket_dev, int bar,
        enum tpu_common_security_level group)
{
 struct tpuv4_device_data *tpuv4_data = gasket_dev->cb_data;
 switch (bar) {
 case 0:
  return tpuv4_data->bar0_get_region_count(group);
 case 2:
  return tpuv4_data->bar2_get_region_count(group);
 default:
  return 0;
 }
}
static const struct gasket_mappable_region *
tpuv4_get_bar_regions(struct gasket_dev *gasket_dev, int bar,
      enum tpu_common_security_level group)
{
 struct tpuv4_device_data *tpuv4_data = gasket_dev->cb_data;
 switch (bar) {
 case 0:
  return tpuv4_data->bar0_get_regions(group);
 case 2:
  return tpuv4_data->bar2_get_regions(group);
 default:
  return NULL;
 }
}
int tpuv4_get_mappable_regions_cb(
 struct gasket_filp_data *filp_data, int bar_index,
 struct gasket_mappable_region **mappable_regions,
 int *num_mappable_regions)
{
 return tpu_common_get_mappable_regions(
  filp_data->gasket_dev, bar_index, tpuv4_get_bar_region_count,
  tpuv4_get_bar_regions, mappable_regions, num_mappable_regions);
}
static int reset_complete(struct gasket_dev *gasket_dev, bool log_not_complete)
{
 const struct asic_fw_state_offsets *state_offsets;
 const struct asic_fw_reinit_reset_offsets *reinit_reset_offsets;
 const struct tpuv4common_pcie_flr_status_offsets *pcie_flr_status_offsets;
 struct tpuv4_device_data *tpuv4_data = gasket_dev->cb_data;
 state_offsets = tpuv4_data->firmware_state_offsets;
 reinit_reset_offsets = tpuv4_data->firmware_reinit_reset_offsets;
 pcie_flr_status_offsets = tpuv4_data->pcie_flr_status_offsets;
 return tpuv4common_reset_complete(gasket_dev, tpuv4_data->software_csr_bar_index,
      reinit_reset_offsets->chip_reset,
      tpuv4_data->reset_accepted_value,
      state_offsets->firmware_status,
      tpuv4_data->chip_init_done_value,
      pcie_flr_status_offsets->pcie_flr_status,
                              0, log_not_complete);
}
int tpuv4_reset(struct gasket_dev *gasket_dev, uint type)
{
 const struct asic_fw_reinit_reset_offsets *reinit_reset_offsets;
 struct tpuv4_device_data *tpuv4_data = gasket_dev->cb_data;
 if (type == tpuv4_data->reinit_reset_value) {
  reinit_reset_offsets = tpuv4_data->firmware_reinit_reset_offsets;
  return tpu_common_reinit_reset(gasket_dev, tpuv4_data->software_csr_bar_index,
         tpuv4_data->reset_retry_value,
         tpuv4_data->reset_delay_value,
         reset_complete,
         reinit_reset_offsets->chip_reset,
         tpuv4_data->reinit_reset_value,
         tpuv4_data->reset_accepted_value);
 } else {
  gasket_log_error(gasket_dev, "invalid reset type specified: %u",
     type);
  return -EINVAL;
 }
}
