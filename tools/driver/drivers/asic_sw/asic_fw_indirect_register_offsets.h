/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_ASIC_SW_ASIC_FW_INDIRECT_REGISTER_OFFSETS_H_
#define _DRIVERS_ASIC_SW_ASIC_FW_INDIRECT_REGISTER_OFFSETS_H_ 
#include "drivers/gasket/gasket_types.h"
struct asic_fw_indirect_register_offsets {
 uint64 indirect_accessor_address;
 uint64 indirect_accessor_control;
 uint64 indirect_accessor_status;
 uint64 indirect_accessor_value;
 uint64 indirect_accessor_version;
};
#endif
