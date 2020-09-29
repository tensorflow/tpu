/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#include "gasket_sysfs.h"
#include "gasket_logging.h"
struct gasket_sysfs_mapping {
 struct device *device;
 struct device *legacy_device;
 struct gasket_dev *gasket_dev;
 struct gasket_sysfs_attribute *attributes;
 int attribute_count;
 struct mutex mutex;
};
static struct gasket_sysfs_mapping dev_mappings[GASKET_SYSFS_NUM_MAPPINGS];
static struct gasket_sysfs_mapping *get_mapping(struct device *device)
{
 int i;
 if (device == NULL) {
  gasket_nodev_error("Received NULL device!");
  return NULL;
 }
 for (i = 0; i < GASKET_SYSFS_NUM_MAPPINGS; i++) {
  mutex_lock(&dev_mappings[i].mutex);
  if ((dev_mappings[i].device == device ||
       dev_mappings[i].legacy_device == device)) {
   mutex_unlock(&dev_mappings[i].mutex);
   return &dev_mappings[i];
  }
  mutex_unlock(&dev_mappings[i].mutex);
 }
 gasket_nodev_info("Mapping to device %s not found.", device->kobj.name);
 return NULL;
}
static void remove_entries(struct device *device, struct device *legacy_device,
      struct device_attribute *entries, int num_entries)
{
 int i;
 for (i = 0; i < num_entries; ++i) {
  device_remove_file(device, &entries[i]);
  if (legacy_device)
   device_remove_file(legacy_device, &entries[i]);
 }
}
void gasket_sysfs_init(void)
{
 int i;
 for (i = 0; i < GASKET_SYSFS_NUM_MAPPINGS; i++) {
  dev_mappings[i].device = NULL;
  mutex_init(&dev_mappings[i].mutex);
 }
}
int gasket_sysfs_create_mapping(
 struct device *device, struct gasket_dev *gasket_dev)
{
 struct gasket_sysfs_mapping *mapping;
 int map_idx, ret = 0;
 static DEFINE_MUTEX(function_mutex);
 mutex_lock(&function_mutex);
 mapping = get_mapping(device);
 if (mapping != NULL) {
  gasket_log_error(gasket_dev,
   "sysfs mapping already exists for device %s.",
   device->kobj.name);
  ret = -EINVAL;
  goto unlock_function;
 }
 for (map_idx = 0; map_idx < GASKET_SYSFS_NUM_MAPPINGS; ++map_idx) {
  mutex_lock(&dev_mappings[map_idx].mutex);
  if (dev_mappings[map_idx].device == NULL)
   break;
  mutex_unlock(&dev_mappings[map_idx].mutex);
 }
 if (map_idx == GASKET_SYSFS_NUM_MAPPINGS) {
  gasket_log_error(
   gasket_dev, "All mappings have been exhausted!");
  mutex_unlock(&function_mutex);
  ret = -ENOMEM;
  goto unlock_function;
 }
 gasket_log_info(gasket_dev,
  "Creating sysfs mapping for device %s.", device->kobj.name);
 mapping = &dev_mappings[map_idx];
 mapping->device = device;
 mapping->legacy_device = gasket_dev->legacy_device;
 mapping->gasket_dev = gasket_dev;
 mapping->attributes = kzalloc(
  GASKET_SYSFS_MAX_NODES * sizeof(*(mapping->attributes)),
  GFP_KERNEL);
 mapping->attribute_count = 0;
 if (!mapping->attributes) {
  gasket_log_error(gasket_dev,
     "Unable to allocate sysfs attribute array.");
  ret = -ENOMEM;
 }
 mutex_unlock(&mapping->mutex);
unlock_function:
 mutex_unlock(&function_mutex);
 return ret;
}
EXPORT_SYMBOL(gasket_sysfs_create_mapping);
int gasket_sysfs_create_entries(
 struct device *device, const struct gasket_sysfs_attribute *attrs)
{
 int i;
 int ret = 0;
 struct gasket_sysfs_mapping *mapping = get_mapping(device);
 if (mapping == NULL)
  return -EINVAL;
 mutex_lock(&mapping->mutex);
 for (i = 0; strcmp(attrs[i].attr.attr.name, GASKET_ARRAY_END_MARKER);
  i++) {
  if (mapping->attribute_count == GASKET_SYSFS_MAX_NODES) {
   gasket_log_error(mapping->gasket_dev,
    "Maximum number of sysfs nodes reached for device.");
   ret = -ENOMEM;
   goto out;
  }
  ret = device_create_file(device, &attrs[i].attr);
  if (ret) {
   gasket_log_error(mapping->gasket_dev,
    "Unable to create sysfs entry %s; rc: %d",
    attrs[i].attr.attr.name, ret);
   goto out;
  }
  if (mapping->legacy_device) {
   ret = device_create_file(mapping->legacy_device,
    &(attrs[i].attr));
   if (ret) {
    gasket_log_error(mapping->gasket_dev,
     "Unable to create legacy sysfs entries; rc: %d",
     ret);
    goto out;
   }
  }
  mapping->attributes[mapping->attribute_count] = attrs[i];
  ++mapping->attribute_count;
 }
out:
 mutex_unlock(&mapping->mutex);
 return ret;
}
EXPORT_SYMBOL(gasket_sysfs_create_entries);
int gasket_sysfs_remove_entries(
 struct device *device, const struct gasket_sysfs_attribute *attrs)
{
 int i, j, num_entries = 0, current_entry = 0;
 struct gasket_sysfs_mapping *mapping = get_mapping(device);
 struct device_attribute *entries;
 const char *attr_name;
 bool entry_found;
 if (mapping == NULL)
  return -EINVAL;
 for (i = 0; strcmp(attrs[i].attr.attr.name, GASKET_ARRAY_END_MARKER);
      i++)
  num_entries++;
 entries = kcalloc(num_entries, sizeof(*entries), GFP_KERNEL);
 mutex_lock(&mapping->mutex);
 for (i = 0; strcmp(attrs[i].attr.attr.name, GASKET_ARRAY_END_MARKER);
      i++) {
  attr_name = attrs[i].attr.attr.name;
  entry_found = false;
  for (j = 0; j < GASKET_SYSFS_MAX_NODES; j++) {
   if (attr_name &&
       !strcmp(mapping->attributes[j].attr.attr.name,
        attr_name)) {
    entries[current_entry] =
      mapping->attributes[j].attr;
    entry_found = true;
    break;
   }
  }
  if (!entry_found) {
   gasket_log_warn(
    mapping->gasket_dev, "Sysfs attr %s not found.",
    attr_name);
  } else
   current_entry++;
 }
 mapping->attribute_count -= current_entry;
 mutex_unlock(&mapping->mutex);
 remove_entries(mapping->device, mapping->legacy_device, entries,
         current_entry);
 kfree(entries);
 return 0;
}
EXPORT_SYMBOL(gasket_sysfs_remove_entries);
void gasket_sysfs_remove_mapping(struct device *device)
{
 struct gasket_sysfs_mapping *mapping = get_mapping(device);
 struct device_attribute *files_to_remove;
 int i, num_files_to_remove;
 if (mapping == NULL)
  return;
 mutex_lock(&mapping->mutex);
 mutex_unlock(&mapping->mutex);
 gasket_log_info(mapping->gasket_dev,
  "Removing Gasket sysfs mapping, device %s",
  mapping->device->kobj.name);
 num_files_to_remove = mapping->attribute_count;
 files_to_remove = kcalloc(
   num_files_to_remove, sizeof(*files_to_remove),
   GFP_KERNEL);
 for (i = 0; i < num_files_to_remove; i++)
  files_to_remove[i] = mapping->attributes[i].attr;
 remove_entries(mapping->device, mapping->legacy_device,
         files_to_remove, num_files_to_remove);
 kfree(files_to_remove);
 mutex_lock(&mapping->mutex);
 kfree(mapping->attributes);
 mapping->attributes = NULL;
 mapping->attribute_count = 0;
 mapping->device = NULL;
 mapping->legacy_device = NULL;
 mapping->gasket_dev = NULL;
 mutex_unlock(&mapping->mutex);
}
EXPORT_SYMBOL(gasket_sysfs_remove_mapping);
struct gasket_dev *gasket_sysfs_get_device_data(struct device *device)
{
 struct gasket_sysfs_mapping *mapping = get_mapping(device);
 if (mapping == NULL)
  return NULL;
 return mapping->gasket_dev;
}
EXPORT_SYMBOL(gasket_sysfs_get_device_data);
struct gasket_sysfs_attribute *gasket_sysfs_get_attr(
 struct device *device, struct device_attribute *attr)
{
 int i;
 int num_attrs;
 struct gasket_sysfs_mapping *mapping = get_mapping(device);
 struct gasket_sysfs_attribute *attrs = NULL;
 if (mapping == NULL)
  return NULL;
 attrs = mapping->attributes;
 num_attrs = mapping->attribute_count;
 for (i = 0; i < num_attrs; ++i) {
  if (!strcmp(attrs[i].attr.attr.name, attr->attr.name))
   return &attrs[i];
 }
 gasket_log_error(mapping->gasket_dev,
  "Unable to find match for device_attribute %s",
  attr->attr.name);
 return NULL;
}
EXPORT_SYMBOL(gasket_sysfs_get_attr);
ssize_t gasket_sysfs_register_show(
 struct device *device, struct device_attribute *attr, char *buf)
{
 ulong reg_address, reg_bar, reg_value;
 struct gasket_sysfs_mapping *mapping;
 struct gasket_dev *gasket_dev;
 struct gasket_sysfs_attribute *gasket_attr;
 mapping = get_mapping(device);
 if (mapping == NULL) {
  gasket_nodev_info("Device driver may have been removed.");
  return 0;
 }
 gasket_dev = mapping->gasket_dev;
 if (gasket_dev == NULL) {
  gasket_nodev_info("Device driver may have been removed.");
  return 0;
 }
 gasket_attr = gasket_sysfs_get_attr(device, attr);
 if (gasket_attr == NULL) {
  return 0;
 }
 reg_address = gasket_attr->data.bar_address.offset;
 reg_bar = gasket_attr->data.bar_address.bar;
 reg_value = gasket_dev_read_64(gasket_dev, reg_bar, reg_address);
 return snprintf(buf, PAGE_SIZE, "0x%lX\n", reg_value);
}
EXPORT_SYMBOL(gasket_sysfs_register_show);
ssize_t gasket_sysfs_register_store(
 struct device *device, struct device_attribute *attr, const char *buf,
 size_t count)
{
 ulong parsed_value = 0;
 struct gasket_sysfs_mapping *mapping;
 struct gasket_dev *gasket_dev;
 struct gasket_sysfs_attribute *gasket_attr;
 if (count < 3 || buf[0] != '0' || buf[1] != 'x') {
  gasket_nodev_error(
   "sysfs register write format: \"0x<hex value>\".");
  return -EINVAL;
 }
 if (kstrtoul(buf, 16, &parsed_value) != 0) {
  gasket_nodev_error(
   "Unable to parse input as 64-bit hex value: %s.", buf);
  return -EINVAL;
 }
 mapping = get_mapping(device);
 if (mapping == NULL) {
  gasket_nodev_info("Device driver may have been removed.");
  return 0;
 }
 gasket_dev = mapping->gasket_dev;
 if (gasket_dev == NULL) {
  gasket_nodev_info("Device driver may have been removed.");
  return 0;
 }
 gasket_attr = gasket_sysfs_get_attr(device, attr);
 if (gasket_attr == NULL) {
  return count;
 }
 gasket_dev_write_64(gasket_dev, parsed_value,
  gasket_attr->data.bar_address.bar,
  gasket_attr->data.bar_address.offset);
 if (gasket_attr->write_callback != NULL)
  gasket_attr->write_callback(
   gasket_dev, gasket_attr, parsed_value);
 return count;
}
EXPORT_SYMBOL(gasket_sysfs_register_store);
