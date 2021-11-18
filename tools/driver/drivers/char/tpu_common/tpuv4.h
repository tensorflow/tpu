/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "drivers/asic_sw/asic_fw_device_owner_offsets.h"
#include "drivers/asic_sw/asic_fw_indirect_register_offsets.h"
#include "drivers/asic_sw/asic_fw_state_offsets.h"
#include "drivers/asic_sw/asic_fw_version_offsets.h"
#include "drivers/asic_sw/asic_fw_reinit_reset_offsets.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_mgr_error_loperf_offsets.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_pcie_flr_status_offsets.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_userspace_lst_port_indirect_offsets.h"
#include "drivers/char/tpu_common/tpu_common.h"
#include "drivers/char/tpu_common/tpuv4common.h"
#include "drivers/gasket/gasket_core.h"
#define TPUV4_DRIVER_VERSION "0.0.1"
#define TPUV4_NUM_MSIX_INTERRUPTS 128
struct tpuv4_setup_data {
 int software_csr_bar_index;
 int reinit_reset_value;
 int reset_accepted_value;
 int chip_init_done_value;
 int reset_retry_value;
 int reset_delay_value;
 const struct tpuv4common_mgr_error_loperf_offsets *(*error_offsets_get)(int index);
 const struct asic_fw_device_owner_offsets *(*device_owner_offsets_get)(
  int index);
 const struct asic_fw_version_offsets *(*firmware_version_offsets_get)(
  int index);
 const struct asic_fw_indirect_register_offsets *(
  *indirect_register_offsets_get)(int index);
 int (*userspace_lst_port_indirect_offsets_count)(void);
 const struct tpuv4common_userspace_lst_port_indirect_offsets *(
  *userspace_lst_port_indirect_offsets_get)(int index);
 int (*bar0_get_region_count)(enum tpu_common_security_level group);
 const struct gasket_mappable_region *(*bar0_get_regions)(
  enum tpu_common_security_level group);
 int (*bar2_get_region_count)(enum tpu_common_security_level group);
 const struct gasket_mappable_region *(*bar2_get_regions)(
  enum tpu_common_security_level group);
 const struct asic_fw_state_offsets *(*firmware_state_offsets_get)(int index);
 const struct asic_fw_reinit_reset_offsets *(
  *firmware_reinit_reset_offsets_get)(int index);
 const struct tpuv4common_pcie_flr_status_offsets *(
  *pcie_flr_status_offsets_get)(int index);
};
int tpuv4_device_open_cb(struct gasket_filp_data *filp_data, struct file *file);
int tpuv4_device_close_cb(struct gasket_filp_data *filp_data, struct file *file);
enum gasket_status tpuv4_get_status(struct gasket_dev *gasket_dev);
int tpuv4_initialize(struct gasket_dev *gasket_dev,
     const struct tpuv4_setup_data *setup_data);
int tpuv4_remove_dev_cb(struct gasket_dev *gasket_dev);
int tpuv4_get_mappable_regions_cb(
 struct gasket_filp_data *filp_data, int bar_index,
 struct gasket_mappable_region **mappable_regions,
 int *num_mappable_regions);
int tpuv4_reset(struct gasket_dev *gasket_dev, uint type);
