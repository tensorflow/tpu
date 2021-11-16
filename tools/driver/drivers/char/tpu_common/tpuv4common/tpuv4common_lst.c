/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_lst.h"
#include <linux/export.h>
#include "drivers/asic_sw/asic_sw_clock.h"
#include "drivers/asic_sw/asic_sw_firmware_indirect_registers.h"
#include "drivers/char/tpu_common/tpuv4common/tpuv4common_userspace_lst_port_indirect_accessor.h"
#include "drivers/gasket/gasket_logging.h"
int tpuv4common_lst_disable_lst_data_links(
 struct gasket_dev *gasket_dev,
 const struct firmware_indirect *firmware_indirect,
 const struct tpuv4common_userspace_lst_port_indirect_offsets
  *lst_userspace_port_indirect_offsets,
 int disable_poll_count, uint64 disable_poll_interval_msecs)
{
 int i, ret;
 uint64 data_link_request_reg;
 uint64 data_link_status_reg;
 data_link_request_reg = 0;
 set_tpuv4common_data_link_layer_request_request(
  &data_link_request_reg,
  kTpuv4commonDataLinkLayerRequestRequestValueGoDown);
 ret = asic_sw_firmware_indirect_write_64(
  gasket_dev, firmware_indirect,
  lst_userspace_port_indirect_offsets->data_link_layer_request,
  data_link_request_reg);
 if (ret) {
  gasket_log_error(
   gasket_dev,
   "Failed to set indirect data_link_layer_request to go down");
  return ret;
 }
 for (i = 0; i < disable_poll_count; i++) {
  ret = asic_sw_firmware_indirect_read_64(
   gasket_dev, firmware_indirect,
   lst_userspace_port_indirect_offsets
    ->data_link_layer_status,
   &data_link_status_reg);
  if (ret) {
   gasket_log_error(
    gasket_dev,
    "Failed to read indirect data_link_layer_status");
   return ret;
  }
  if (tpuv4common_data_link_layer_status_status(data_link_status_reg) ==
      kTpuv4commonDataLinkLayerStatusStatusValueDown) {
   return 0;
  }
  asic_sw_sleep_for_msecs(disable_poll_interval_msecs);
 }
 gasket_log_error(gasket_dev,
    "Timed out waiting for data_link_layer to go down");
 return 1;
}
EXPORT_SYMBOL(tpuv4common_lst_disable_lst_data_links);
