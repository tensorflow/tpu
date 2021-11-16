/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "tpu_common.h"
#include <linux/google/asic_fw_device_owner_accessor.h>
#include "drivers/gasket/gasket_logging.h"
#include "drivers/gasket/gasket_sysfs.h"
#define TPU_COMMON_COMMON_VERSION "1.0.0"
static ssize_t sysfs_show(struct device *device, struct device_attribute *attr,
     char *buf);
static ssize_t sysfs_store(struct device *device, struct device_attribute *attr,
      const char *buf, size_t count);
static int tpu_common_set_fw_device_owend(struct gasket_dev *gasket_dev,
           struct tpu_common_device_data *device_data,
           unsigned long device_owned_value);
enum sysfs_attribute_type {
 ATTR_RESET_ON_CLOSE,
 ATTR_RESET_ON_OPEN,
};
struct gasket_sysfs_attribute sysfs_attrs[] = {
 GASKET_SYSFS_RW(reset_on_close, sysfs_show, sysfs_store,
   ATTR_RESET_ON_CLOSE),
 GASKET_SYSFS_RW(reset_on_open, sysfs_show, sysfs_store,
   ATTR_RESET_ON_OPEN),
 GASKET_END_OF_ATTR_ARRAY
};
int tpu_common_setup_device_data(struct tpu_common_device_data *device_data,
                                 uint device_open_reset_type,
                                 int device_owned_bar,
                                 unsigned long device_owned_offset,
                                 unsigned long device_firmware_version_offset) {
  device_data->reset_on_close = 1;
  device_data->reset_on_open = 1;
  device_data->device_open_reset_type = device_open_reset_type;
  device_data->device_owned_bar = device_owned_bar;
  device_data->device_owned_offset = device_owned_offset;
  device_data->device_owned_value = 0;
  device_data->device_firmware_version_offset = device_firmware_version_offset;
  return 0;
}
EXPORT_SYMBOL(tpu_common_setup_device_data);
int tpu_common_reinit_reset(struct gasket_dev *gasket_dev, int bar_index,
    int retry_count, int retry_delay,
    int (*reset_complete)(struct gasket_dev *gasket_dev,
            bool log_not_complete),
    unsigned long reset_offset,
    unsigned long reset_start_value,
    unsigned long reset_accepted_value)
{
 int ret = 0;
 ulong reset_val;
 uint retry;
 for (retry = 0; retry < retry_count; retry++) {
  ret = (*reset_complete)(gasket_dev,
                          false);
  if (ret != -EBUSY)
   break;
  set_current_state(TASK_UNINTERRUPTIBLE);
  schedule_timeout(msecs_to_jiffies(retry_delay));
 }
 ret = (*reset_complete)(gasket_dev, true);
 if (ret)
  return ret;
 gasket_log_debug(gasket_dev, "Performing reset (type %lu)",
    reset_start_value);
 gasket_dev_write_64(gasket_dev, reset_start_value, bar_index,
       reset_offset);
 gasket_dev->reset_count++;
 for (retry = 0; retry < retry_count; retry++) {
  if (gasket_dev_read_64(gasket_dev, bar_index, reset_offset) ==
      reset_accepted_value)
   break;
  set_current_state(TASK_UNINTERRUPTIBLE);
  schedule_timeout(msecs_to_jiffies(retry_delay));
 }
 reset_val = gasket_dev_read_64(gasket_dev, bar_index, reset_offset);
 if (reset_val != reset_accepted_value) {
  gasket_log_error(
   gasket_dev,
   "Device did not accept reset. Reset register value: %lu",
   reset_val);
  return -EBUSY;
 }
 for (retry = 0; retry < retry_count; retry++) {
  ret = (*reset_complete)(gasket_dev,
                          false);
  if (ret != -EBUSY)
   break;
  set_current_state(TASK_UNINTERRUPTIBLE);
  schedule_timeout(msecs_to_jiffies(retry_delay));
 }
 ret = (*reset_complete)(gasket_dev, true);
 if (ret)
  gasket_log_error(gasket_dev,
     "Device did not come back from reset.");
 return ret;
}
EXPORT_SYMBOL(tpu_common_reinit_reset);
int tpu_common_device_open(struct gasket_dev *gasket_dev,
   struct tpu_common_device_data *device_data,
   unsigned long fw_device_owned_value)
{
 int ret;
 if (gasket_dev->ownership.is_owned) {
  if (gasket_dev->ownership.owner != current->tgid) {
   gasket_log_error(
    gasket_dev,
    "Process %u is opening a node held by %u.",
    current->tgid, gasket_dev->ownership.owner);
   return -EPERM;
  }
  return 0;
 }
 ret = 0;
 if (device_data->reset_on_open) {
  ret = gasket_reset_nolock(gasket_dev,
       device_data->device_open_reset_type);
 } else {
   gasket_log_warn(gasket_dev, "skipping reset on open");
 }
 if (ret) {
  gasket_log_error(gasket_dev, "Failed to reset device");
  return ret;
 }
 if (!gasket_dev->ownership.is_owned)
  ret = tpu_common_set_fw_device_owend(gasket_dev, device_data,
        fw_device_owned_value);
 return ret;
}
EXPORT_SYMBOL(tpu_common_device_open);
int tpu_common_get_mappable_regions(
    struct gasket_dev *gasket_dev, int bar_index,
    int (*bar_region_count_cb)(struct gasket_dev *gasket_dev, int bar,
                               enum tpu_common_security_level group),
    const struct gasket_mappable_region *(*get_bar_regions_cb)(
        struct gasket_dev *gasket_dev, int bar,
        enum tpu_common_security_level group),
    struct gasket_mappable_region **mappable_regions,
    int *num_mappable_regions) {
  uint64_t region_size;
  enum tpu_common_security_level target_security_level;
  if (capable(CAP_SYS_ADMIN))
    target_security_level = TPU_COMMON_SECURITY_LEVEL_ROOT;
  else
    target_security_level = TPU_COMMON_SECURITY_LEVEL_USER;
  *num_mappable_regions =
      bar_region_count_cb(gasket_dev, bar_index, target_security_level);
  if (*num_mappable_regions == 0) return 0;
  *mappable_regions =
      kzalloc(sizeof(struct gasket_mappable_region) * (*num_mappable_regions),
              GFP_KERNEL);
  if (*mappable_regions == NULL) {
    gasket_log_error(gasket_dev, "Unable to alloc mappable region block!");
    *num_mappable_regions = 0;
    return -ENOMEM;
  }
  region_size = (*num_mappable_regions) * sizeof(struct gasket_mappable_region);
  memcpy(*mappable_regions,
         get_bar_regions_cb(gasket_dev, bar_index, target_security_level),
         region_size);
  return 0;
}
EXPORT_SYMBOL(tpu_common_get_mappable_regions);
int tpu_common_sysfs_setup(struct gasket_dev *gasket_dev)
{
 int ret;
 ret = gasket_sysfs_create_entries(&gasket_dev->accel_dev.dev,
       sysfs_attrs);
 if (ret)
  gasket_log_error(gasket_dev, "Unable to setup tpu_common sysfs!");
 return ret;
}
EXPORT_SYMBOL(tpu_common_sysfs_setup);
static ssize_t sysfs_show(struct device *device, struct device_attribute *attr,
     char *buf)
{
 int ret;
 struct gasket_dev *gasket_dev;
 struct gasket_sysfs_attribute *gasket_attr;
 struct tpu_common_device_data *tpu_common_data;
 enum sysfs_attribute_type type;
 gasket_dev = gasket_sysfs_get_device_data(device);
 if (gasket_dev == NULL)
  return 0;
 tpu_common_data = gasket_dev->cb_data;
 gasket_attr = gasket_sysfs_get_attr(device, attr);
 if (gasket_attr == NULL)
  return 0;
 type = (enum sysfs_attribute_type)gasket_attr->data.attr_type;
 switch (type) {
 case ATTR_RESET_ON_CLOSE:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    tpu_common_data->reset_on_close);
  break;
 case ATTR_RESET_ON_OPEN:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    tpu_common_data->reset_on_open);
  break;
 default:
  gasket_log_error(gasket_dev, "Unknown attribute: %s",
     attr->attr.name);
  ret = 0;
  break;
 }
 return ret;
}
static ssize_t sysfs_store(struct device *device, struct device_attribute *attr,
      const char *buf, size_t count)
{
 int ret;
 int parse_buffer;
 struct gasket_dev *gasket_dev;
 struct tpu_common_device_data *tpu_common_data;
 struct gasket_sysfs_attribute *gasket_attr;
 enum sysfs_attribute_type type;
 gasket_dev = gasket_sysfs_get_device_data(device);
 if (gasket_dev == NULL)
  return 0;
 tpu_common_data = gasket_dev->cb_data;
 gasket_attr = gasket_sysfs_get_attr(device, attr);
 if (gasket_attr == NULL)
  return 0;
 type = (enum sysfs_attribute_type)gasket_attr->data.attr_type;
 switch (type) {
 case ATTR_RESET_ON_CLOSE:
  ret = kstrtoint(buf, 10, &parse_buffer);
  if (ret) {
   gasket_log_error(gasket_dev,
      "Invalid reset_on_close argument: %s",
      buf);
  } else {
   tpu_common_data->reset_on_close = parse_buffer ? 1 : 0;
   ret = count;
  }
  break;
 case ATTR_RESET_ON_OPEN:
  ret = kstrtoint(buf, 10, &parse_buffer);
  if (ret) {
   gasket_log_error(gasket_dev,
      "Invalid reset_on_open argument: %s",
      buf);
  } else {
   tpu_common_data->reset_on_open = parse_buffer ? 1 : 0;
   ret = count;
  }
  break;
 default:
  gasket_log_error(gasket_dev, "Unknown attribute: %s",
     attr->attr.name);
  ret = 0;
  break;
 }
 return ret;
}
int tpu_common_set_fw_device_owend(struct gasket_dev *gasket_dev,
    struct tpu_common_device_data *device_data,
    unsigned long device_owned_value)
{
 int ret;
 unsigned long device_owner_reg;
 device_owner_reg =
  gasket_dev_read_64(gasket_dev, device_data->device_owned_bar,
       device_data->device_owned_offset);
 if (device_owner_reg) {
  gasket_log_error(gasket_dev,
     "Trying to set ownership while owned by: %ld",
     device_owner_reg);
 }
 ret = set_asic_fw_device_owner_value(&device_owner_reg,
          device_owned_value);
 if (ret) {
  gasket_log_error(gasket_dev,
     "Failed to encode owned value %lu , ret %d",
     device_owned_value, ret);
  return ret;
 }
 gasket_dev_write_64(gasket_dev, device_owner_reg,
       device_data->device_owned_bar,
       device_data->device_owned_offset);
 device_data->device_owned_value = device_owned_value;
 return 0;
}
int tpu_common_clear_fw_device_owned(struct gasket_dev *gasket_dev,
      struct tpu_common_device_data *device_data)
{
 int ret;
 unsigned long device_owner_reg;
 device_owner_reg = 0;
 device_owner_reg =
  gasket_dev_read_64(gasket_dev, device_data->device_owned_bar,
       device_data->device_owned_offset);
 if (device_owner_reg == device_data->device_owned_value) {
  ret = set_asic_fw_device_owner_value(&device_owner_reg, 0);
  if (ret) {
   gasket_log_error(
    gasket_dev,
    "Failed to encode 0 in device_owner_reg, ret %d",
    ret);
   return ret;
  }
  gasket_dev_write_64(gasket_dev, device_owner_reg,
        device_data->device_owned_bar,
        device_data->device_owned_offset);
 } else if (device_owner_reg == 0) {
  gasket_log_error(
   gasket_dev,
   "Trying to clear firmware device owned when not owned");
 } else {
  gasket_log_error(
   gasket_dev,
   "Trying to clear firmware device owned when owned by someone else. self: %lu owner: %lu",
   device_data->device_owned_value, device_owner_reg)
 }
 device_data->device_owned_value = 0;
 return 0;
}
EXPORT_SYMBOL(tpu_common_clear_fw_device_owned);
int tpu_common_get_hardware_revision(struct gasket_dev *gasket_dev) {
  return gasket_dev->pci_dev->revision;
}
EXPORT_SYMBOL(tpu_common_get_hardware_revision);
int tpu_common_get_firmware_version_cb(struct gasket_dev *gasket_dev,
                                       unsigned int *major, unsigned int *minor,
                                       unsigned int *point,
                                       unsigned int *subpoint) {
  struct tpu_common_device_data *device_data;
  unsigned long version;
  device_data = gasket_dev->cb_data;
  if (!device_data->device_firmware_version_offset) return -EINVAL;
  version = gasket_dev_read_64(gasket_dev, device_data->device_owned_bar,
                               device_data->device_firmware_version_offset);
  *major = (version >> 48);
  *minor = (version >> 32) & 0xffff;
  *point = (version >> 16) & 0xffff;
  *subpoint = version & 0xffff;
  return 0;
}
EXPORT_SYMBOL(tpu_common_get_firmware_version_cb);
MODULE_DESCRIPTION("Google tpu_common Common Library");
MODULE_VERSION(TPU_COMMON_COMMON_VERSION);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Googler <noreply@google.com>");
