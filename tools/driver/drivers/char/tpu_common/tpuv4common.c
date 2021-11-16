/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpuv4common.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_lst.h"
#include "drivers/gasket/gasket_logging.h"
#define TPU_V4_COMMON_COMMON_VERSION "1.0.0"
#define INDIRECT_REGISTERS_BAR 2
#define TPU_V4_COMMON_LST_DISABLE_POLL_COUNT 100
#define TPU_V4_COMMON_LST_POLL_INTERVAL_MSECS 100
int tpuv4common_reset_complete(struct gasket_dev *gasket_dev, int bar_index,
         unsigned long reset_offset,
         unsigned long reset_accepted_value,
         unsigned long firmware_state_offset,
         unsigned long firmware_state_ready_value,
         unsigned long pcie_flr_status_offset,
         unsigned long pcie_flr_done_value, bool log_not_complete)
{
 int ret = 0;
 ulong flr_status;
 ulong init_val;
 ulong reset_val;
 init_val = gasket_dev_read_64(gasket_dev, bar_index,
          firmware_state_offset);
 reset_val = gasket_dev_read_64(gasket_dev, bar_index, reset_offset);
 flr_status = gasket_dev_read_64(gasket_dev, bar_index,
     pcie_flr_status_offset);
 if (flr_status != pcie_flr_done_value) {
  if (log_not_complete)
   gasket_log_error(
    gasket_dev,
    "Device is currently busy. FLR status: %lu",
    flr_status);
  return -EBUSY;
 }
 if ((init_val != firmware_state_ready_value) ||
     (reset_val != reset_accepted_value)) {
  if (log_not_complete)
   gasket_log_error(
    gasket_dev,
    "Device is currently busy. Firmware state value: %lu; reset register value %lu",
    init_val, reset_val);
  return -EBUSY;
 }
 return ret;
}
EXPORT_SYMBOL(tpuv4common_reset_complete);
static void disable_lst(struct gasket_dev *gasket_dev,
   struct tpuv4common_device_data *device_data)
{
 int i, ret;
 if (device_data->userspace_lst_port_indirect_offsets_count == 0 ||
     device_data->userspace_lst_port_indirect_offsets_get == NULL) {
  return;
 }
 for (i = 0; i < device_data->userspace_lst_port_indirect_offsets_count;
      i++) {
  ret = tpuv4common_lst_disable_lst_data_links(
   gasket_dev, &device_data->firmware_indirect,
   device_data->userspace_lst_port_indirect_offsets_get(i),
   TPU_V4_COMMON_LST_DISABLE_POLL_COUNT,
   TPU_V4_COMMON_LST_POLL_INTERVAL_MSECS);
  if (ret) {
   gasket_log_error(gasket_dev,
      "Error while disabling lst[%d]: %d", i,
      ret);
  }
 }
}
int tpuv4common_device_close(struct gasket_dev *gasket_dev,
       struct tpuv4common_device_data *device_data, int bar_index,
       unsigned long fatal_error_status_offset, int reset_type)
{
 unsigned long error_status;
 error_status = gasket_dev_read_64(gasket_dev, bar_index,
       fatal_error_status_offset);
 if (error_status) {
  gasket_dev->status = GASKET_STATUS_DEAD;
  gasket_log_error(gasket_dev, "non-zero error_status: 0x%lx",
     error_status);
  disable_lst(gasket_dev, device_data);
 } else {
  if (device_data->tpu_common_data.reset_on_close) {
   gasket_reset_nolock(gasket_dev, reset_type);
  } else {
   gasket_log_warn(gasket_dev, "skipping reset on close");
  }
 }
 tpu_common_clear_fw_device_owned(gasket_dev, &(device_data->tpu_common_data));
 return 0;
}
EXPORT_SYMBOL(tpuv4common_device_close);
int tpuv4common_device_open(struct gasket_dev *gasket_dev,
      struct tpuv4common_device_data *device_data)
{
 return tpu_common_device_open(gasket_dev, &(device_data->tpu_common_data),
                                    current->tgid);
}
EXPORT_SYMBOL(tpuv4common_device_open);
int tpuv4common_setup_device_data(
 struct tpuv4common_device_data *device_data, uint device_open_reset_type,
 int device_owner_bar, unsigned long device_owner_offset,
 unsigned long device_firmware_version_offsets,
 const struct asic_fw_indirect_register_offsets
  *indirect_register_offsets,
 int userspace_lst_port_indirect_offsets_count,
 const struct tpuv4common_userspace_lst_port_indirect_offsets *(
  *userspace_lst_port_indirect_offsets_get)(int index))
{
 int ret;
 ret = tpu_common_setup_device_data(&(device_data->tpu_common_data),
     device_open_reset_type,
     device_owner_bar, device_owner_offset,
     device_firmware_version_offsets);
 device_data->firmware_indirect.offsets = indirect_register_offsets;
 device_data->firmware_indirect.bar = INDIRECT_REGISTERS_BAR;
 device_data->userspace_lst_port_indirect_offsets_count =
  userspace_lst_port_indirect_offsets_count;
 device_data->userspace_lst_port_indirect_offsets_get =
  userspace_lst_port_indirect_offsets_get;
 return ret;
}
EXPORT_SYMBOL(tpuv4common_setup_device_data);
MODULE_DESCRIPTION("Google TPU_V4_COMMON Common Library");
MODULE_VERSION(TPU_V4_COMMON_COMMON_VERSION);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Googler <noreply@google.com>");
