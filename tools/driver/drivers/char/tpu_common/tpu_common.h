/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef __TPU_COMMON_H__
#define __TPU_COMMON_H__ 
#include "drivers/gasket/gasket_core.h"
#define TPU_COMMON_ACCEL_TYPE "tpu_common"
enum tpu_common_security_level {
 TPU_COMMON_SECURITY_LEVEL_ROOT,
 TPU_COMMON_SECURITY_LEVEL_USER,
};
struct tpu_common_device_data {
 int reset_on_close;
 int reset_on_open;
 uint device_open_reset_type;
 int device_owned_bar;
 unsigned long device_owned_offset;
 unsigned long device_owned_value;
 unsigned long device_firmware_version_offset;
};
int tpu_common_setup_device_data(struct tpu_common_device_data *device_data,
         uint device_open_reset_type, int device_owned_bar,
         unsigned long device_owned_offset,
         unsigned long device_firmware_version_offset);
int tpu_common_reinit_reset(struct gasket_dev *gasket_dev, int bar_index,
    int retry_count, int retry_delay,
    int (*reset_complete)(struct gasket_dev *gasket_dev,
            bool log_not_complete),
    unsigned long reset_offset,
    unsigned long reset_start_value,
    unsigned long reset_accepted_value);
int tpu_common_device_open(struct gasket_dev *gasket_dev,
   struct tpu_common_device_data *tpu_common_data,
   unsigned long fw_device_owned_value);
int tpu_common_get_mappable_regions(
 struct gasket_dev *gasket_dev, int bar_index,
 int (*bar_region_count_cb)(struct gasket_dev *gasket_dev, int bar, enum tpu_common_security_level group),
 const struct gasket_mappable_region *(*get_bar_regions_cb)(
  struct gasket_dev *gasket_dev, int bar, enum tpu_common_security_level group),
 struct gasket_mappable_region **mappable_regions,
 int *num_mappable_regions);
int tpu_common_sysfs_setup(struct gasket_dev *gasket_dev);
int tpu_common_clear_fw_device_owned(struct gasket_dev *gasket_dev,
      struct tpu_common_device_data *device_data);
int tpu_common_get_hardware_revision(struct gasket_dev *gasket_dev);
int tpu_common_get_firmware_version_cb(struct gasket_dev *gasket_dev,
 unsigned int *major, unsigned int *minor, unsigned int *point,
 unsigned int *subpoint);
#endif
