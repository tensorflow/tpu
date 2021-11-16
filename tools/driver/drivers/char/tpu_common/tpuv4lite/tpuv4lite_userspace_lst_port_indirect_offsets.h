/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPU_V4_COMMON_USERSPACE_LST_PORT_INDIRECT_OFFSETS_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV4LITE_TPU_V4_COMMON_USERSPACE_LST_PORT_INDIRECT_OFFSETS_H_ 
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_userspace_lst_port_indirect_offsets.h"
#include "drivers/gasket/gasket_types.h"
struct tpuv4lite_userspace_lst_port_indirect_offsets {
 uint64 data_link_layer_request;
 uint64 data_link_layer_status;
 uint64 unused_register_one;
 uint64 unused_register_two;
 uint64 unused_register_three;
 uint64 unused_register_four[2];
 uint64 unused_register_five[2];
 uint64 unused_register_six[2];
 uint64 lock;
 uint64 physical_layer_state;
 uint64 rates;
 uint64 to_mirror[12];
};
int tpuv4lite_userspace_lst_port_indirect_offsets_count(void);
const struct tpuv4common_userspace_lst_port_indirect_offsets *
tpuv4lite_userspace_lst_port_indirect_offsets_get(int index);
#endif
