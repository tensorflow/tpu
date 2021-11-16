/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_ASIC_SW_ASIC_FW_VERSION_OFFSETS_H_
#define _DRIVERS_ASIC_SW_ASIC_FW_VERSION_OFFSETS_H_ 
#include "drivers/gasket/gasket_types.h"
struct asic_fw_version_offsets {
 uint64 changelist;
 uint64 image_info;
 uint64 primary_version;
 uint64 secondary_version;
};
#endif
