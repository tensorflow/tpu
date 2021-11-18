/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef __TPU_V4_COMMON_H__
#define __TPU_V4_COMMON_H__ 
#include "tpu_common.h"
#include "drivers/asic_sw/asic_sw_firmware_indirect_registers.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_userspace_lst_port_indirect_offsets.h"
#include "drivers/gasket/gasket_core.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4common_device_data {
 struct tpu_common_device_data tpu_common_data;
 struct firmware_indirect firmware_indirect;
 int userspace_lst_port_indirect_offsets_count;
 const struct tpuv4common_userspace_lst_port_indirect_offsets *(
  *userspace_lst_port_indirect_offsets_get)(int index);
};
int tpuv4common_reset_complete(struct gasket_dev *gasket_dev, int bar_index,
         unsigned long reset_offset,
         unsigned long reset_accepted_value,
         unsigned long firmware_state_offset,
         unsigned long firmware_state_ready_value,
         unsigned long pcie_flr_status_offset,
         unsigned long pcie_flr_done_value,
         bool log_not_complete);
int tpuv4common_device_close(struct gasket_dev *gasket_dev,
       struct tpuv4common_device_data *device_data, int bar_index,
       unsigned long fatal_error_status_offset, int reset_type);
int tpuv4common_device_open(struct gasket_dev *gasket_dev,
      struct tpuv4common_device_data *device_data);
int tpuv4common_setup_device_data(
 struct tpuv4common_device_data *device_data, uint device_open_reset_type,
 int device_owner_bar, unsigned long device_owner_offset,
 unsigned long device_firmware_version_offsets,
 const struct asic_fw_indirect_register_offsets
  *indirect_register_offsets,
 int userspace_lst_port_indirect_offsets_count,
 const struct tpuv4common_userspace_lst_port_indirect_offsets *(
  *userspace_lst_port_indirect_offsets_get)(int index));
#endif
