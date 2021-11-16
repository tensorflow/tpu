/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef PLATFORMS_ASIC_SW_KERNEL_COMMON_ASIC_SW_FIRMWARE_INDIRECT_REGISTERS_H_
#define PLATFORMS_ASIC_SW_KERNEL_COMMON_ASIC_SW_FIRMWARE_INDIRECT_REGISTERS_H_ 
#include "drivers/asic_sw/asic_fw_indirect_register_offsets.h"
#include "drivers/gasket/gasket_core.h"
#include "drivers/gasket/gasket_types.h"
struct firmware_indirect {
 const struct asic_fw_indirect_register_offsets *offsets;
 int bar;
};
int asic_sw_firmware_indirect_read_64(
 struct gasket_dev *gasket_dev,
 const struct firmware_indirect *firmware_indirect, uint64 address,
 uint64 *read_value);
int asic_sw_firmware_indirect_write_64(
 struct gasket_dev *gasket_dev,
 const struct firmware_indirect *firmware_indirect, uint64 address,
 uint64 write_value);
#endif
