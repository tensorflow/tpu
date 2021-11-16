/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "drivers/asic_sw/asic_sw_firmware_indirect_registers.h"
#include <linux/export.h>
#include "drivers/asic_sw/asic_fw_indirect_accessor.h"
#include "drivers/asic_sw/asic_sw_clock.h"
#include "drivers/gasket/gasket_logging.h"
#define CONTROL_OPERATION_COMPLETE 0
#define CONTROL_READ_ENABLE 1
#define CONTROL_WRITE_ENABLE 1
#define POLL_CONTROL_COUNT 100
#define POLL_CONTROL_INTERVAL_MS 100
static int wait_for_not_busy(struct gasket_dev *gasket_dev,
        const struct firmware_indirect *firmware_indirect)
{
 int retry;
 for (retry = 0; retry < POLL_CONTROL_COUNT; retry++) {
  if (gasket_dev_read_64(gasket_dev, firmware_indirect->bar,
           firmware_indirect->offsets
            ->indirect_accessor_control) ==
      CONTROL_OPERATION_COMPLETE)
   return 0;
  asic_sw_sleep_for_msecs(POLL_CONTROL_INTERVAL_MS);
 }
 gasket_log_warn(gasket_dev,
   "Timed out waiting for idle indirect accessor control");
 return 1;
}
static int block_until_done(struct gasket_dev *gasket_dev,
       const struct firmware_indirect *firmware_indirect)
{
 int retry;
 uint64 raw_status_reg;
 for (retry = 0; retry < POLL_CONTROL_COUNT; retry++) {
  if (gasket_dev_read_64(gasket_dev, firmware_indirect->bar,
           firmware_indirect->offsets
            ->indirect_accessor_control) ==
      CONTROL_OPERATION_COMPLETE) {
   raw_status_reg = gasket_dev_read_64(
    gasket_dev, firmware_indirect->bar,
    firmware_indirect->offsets
     ->indirect_accessor_status);
   switch (asic_fw_indirect_accessor_status_status(
    raw_status_reg)) {
   case (kAsicFwIndirectAccessorStatusStatusValueOk):
    return 0;
   case (kAsicFwIndirectAccessorStatusStatusValueFailed):
    gasket_log_error(
     gasket_dev,
     "Indirect operation failed, chip specific status %d",
     asic_fw_indirect_accessor_status_chip_specific_status(
      raw_status_reg));
    return 1;
   case (kAsicFwIndirectAccessorStatusStatusValueInvalid):
    gasket_log_error(
     gasket_dev,
     "Indirect operation rejected, chip specific status %d",
     asic_fw_indirect_accessor_status_chip_specific_status(
      raw_status_reg));
    return 1;
   default:
    gasket_log_error(
     gasket_dev,
     "Indirect operation returned an invalid status: 0x%llx",
     raw_status_reg);
    return 1;
   }
  }
  asic_sw_sleep_for_msecs(POLL_CONTROL_INTERVAL_MS);
 }
 gasket_log_error(
  gasket_dev,
  "Timed out waiting for idle indirect accessor control");
 return 1;
}
int asic_sw_firmware_indirect_read_64(
 struct gasket_dev *gasket_dev,
 const struct firmware_indirect *firmware_indirect, uint64 address,
 uint64 *read_value)
{
 int ret;
 uint64 control_value = 0;
 ret = wait_for_not_busy(gasket_dev, firmware_indirect);
 if (ret)
  return ret;
 gasket_dev_write_64(
  gasket_dev, address, firmware_indirect->bar,
  firmware_indirect->offsets->indirect_accessor_address);
 set_asic_fw_indirect_accessor_control_read(&control_value,
         CONTROL_READ_ENABLE);
 gasket_dev_write_64(
  gasket_dev, control_value, firmware_indirect->bar,
  firmware_indirect->offsets->indirect_accessor_control);
 ret = block_until_done(gasket_dev, firmware_indirect);
 if (ret)
  return ret;
 *read_value = gasket_dev_read_64(
  gasket_dev, firmware_indirect->bar,
  firmware_indirect->offsets->indirect_accessor_value);
 return 0;
}
EXPORT_SYMBOL(asic_sw_firmware_indirect_read_64);
int asic_sw_firmware_indirect_write_64(
 struct gasket_dev *gasket_dev,
 const struct firmware_indirect *firmware_indirect, uint64 address,
 uint64 write_value)
{
 int ret;
 uint64 control_value = 0;
 ret = wait_for_not_busy(gasket_dev, firmware_indirect);
 if (ret)
  return ret;
 gasket_dev_write_64(
  gasket_dev, address, firmware_indirect->bar,
  firmware_indirect->offsets->indirect_accessor_address);
 gasket_dev_write_64(
  gasket_dev, write_value, firmware_indirect->bar,
  firmware_indirect->offsets->indirect_accessor_value);
 set_asic_fw_indirect_accessor_control_write(&control_value,
          CONTROL_WRITE_ENABLE);
 gasket_dev_write_64(
  gasket_dev, control_value, firmware_indirect->bar,
  firmware_indirect->offsets->indirect_accessor_control);
 return block_until_done(gasket_dev, firmware_indirect);
}
EXPORT_SYMBOL(asic_sw_firmware_indirect_write_64);
