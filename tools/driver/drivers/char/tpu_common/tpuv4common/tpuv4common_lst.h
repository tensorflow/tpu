/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef PLATFORMS_ASIC_SW_KERNEL_TPU_COMMON_TPU_V4_COMMON_COMMON_TPU_V4_COMMON_LST_H_
#define PLATFORMS_ASIC_SW_KERNEL_TPU_COMMON_TPU_V4_COMMON_COMMON_TPU_V4_COMMON_LST_H_ 
#include "drivers/asic_sw/asic_fw_indirect_register_offsets.h"
#include "drivers/asic_sw/asic_sw_firmware_indirect_registers.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_userspace_lst_port_indirect_offsets.h"
#include "drivers/gasket/gasket_core.h"
#include "drivers/gasket/gasket_types.h"
int tpuv4common_lst_disable_lst_data_links(
 struct gasket_dev *gasket_dev,
 const struct firmware_indirect *firmware_indirect,
 const struct tpuv4common_userspace_lst_port_indirect_offsets
  *lst_userspace_port_indirect_offsets,
 int disable_poll_count, uint64 disable_poll_interval_msecs);
#endif
